use std::{cell::RefCell, collections::VecDeque, rc::Rc, sync::Arc};

use crate::{types::audio_frame::RawAudioFrame, CaptureControls, ReadyState};
use crossbeam::channel::Sender;
use pipewire::{
    self as pw,
    context::ContextRc,
    core::CoreRc,
    main_loop::MainLoopRc,
    properties::properties,
    spa::{
        self,
        param::format::{MediaSubtype, MediaType},
        pod::Pod,
        utils::Direction,
    },
    stream::{StreamFlags, StreamListener, StreamRc, StreamState},
    sys::pw_stream_get_nsec,
};

use super::Terminate;

/// Capture format forced on both the desktop and microphone streams so their
/// PCM is sample-aligned for mixing (and matches the Opus encoder's expectation).
const MIX_RATE: u32 = 48_000;
const MIX_CHANNELS: u32 = 2;
/// Cap the microphone jitter buffer at ~0.5s of stereo audio so a stalled
/// desktop stream can never grow it without bound.
const MIC_BUFFER_CAP: usize = (MIX_RATE as usize) * (MIX_CHANNELS as usize) / 2;

/// Shared, single-threaded microphone sample buffer. Both audio streams run on
/// the same PipeWire main loop (one thread), so a plain `Rc<RefCell<..>>` is
/// sufficient and keeps everything on one clock.
type MicBuffer = Rc<RefCell<VecDeque<f32>>>;

#[derive(Clone, Copy, Default)]
struct UserData {
    audio_format: spa::param::audio::AudioInfoRaw,
}

/// Which device a capture stream binds to.
#[derive(Clone, Copy)]
enum Source {
    /// The default sink's monitor (game + voice-chat playback).
    DesktopMonitor,
    /// The default source (the microphone).
    Microphone,
}

struct PipewireState {
    pw_loop: MainLoopRc,
    _pw_context: ContextRc,
    _core: CoreRc,
    _core_listener: pw::core::Listener,
    _stream: StreamRc,
    _stream_listener: StreamListener<UserData>,
    _mic_stream: Option<StreamRc>,
    _mic_listener: Option<StreamListener<UserData>>,
}

pub struct AudioCapture {
    termination_recv: Option<pw::channel::Receiver<Terminate>>,
    pipewire_state: PipewireState,
}

impl AudioCapture {
    pub fn new(
        ready_state: Arc<ReadyState>,
        audio_sender: Sender<RawAudioFrame>,
        termination_recv: pw::channel::Receiver<Terminate>,
        controls: Arc<CaptureControls>,
        include_mic: bool,
    ) -> Result<Self, pw::Error> {
        let pw_loop = MainLoopRc::new(None)?;
        let pw_context = ContextRc::new(&pw_loop, None)?;
        let core = pw_context.connect_rc(None)?;

        let mut core_mut = core.clone();
        let core_listener = Self::setup_core_listener(&mut core_mut);

        // When the mic is enabled, the desktop stream mixes from this buffer.
        let mic_buffer: Option<MicBuffer> =
            include_mic.then(|| Rc::new(RefCell::new(VecDeque::new())));

        // Desktop (sink monitor) stream — the clock master for the mixed track.
        let mut stream =
            Self::create_stream(core.clone(), "waycap-audio-desktop", Source::DesktopMonitor)?;
        let stream_listener = Self::setup_desktop_listener(
            &mut stream,
            ready_state,
            controls,
            audio_sender,
            mic_buffer.clone(),
        )?;
        Self::connect_stream(&mut stream)?;

        // Optional microphone stream feeding the mix buffer.
        let (mic_stream, mic_listener) = if let Some(buf) = mic_buffer.as_ref() {
            let mut mic_stream =
                Self::create_stream(core.clone(), "waycap-audio-mic", Source::Microphone)?;
            let mic_listener = Self::setup_mic_listener(&mut mic_stream, Rc::clone(buf))?;
            Self::connect_stream(&mut mic_stream)?;
            (Some(mic_stream), Some(mic_listener))
        } else {
            (None, None)
        };

        Ok(Self {
            termination_recv: Some(termination_recv),
            pipewire_state: PipewireState {
                pw_loop,
                _pw_context: pw_context,
                _core: core,
                _core_listener: core_listener,
                _stream: stream,
                _stream_listener: stream_listener,
                _mic_stream: mic_stream,
                _mic_listener: mic_listener,
            },
        })
    }

    fn create_stream(core: CoreRc, name: &str, source: Source) -> Result<StreamRc, pw::Error> {
        let mut props = properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::NODE_LATENCY => "1024/48000",
        };
        // For the desktop monitor, ask PipeWire to capture the *default sink's*
        // monitor (game + voice-chat playback) rather than the default source.
        // This is the node-id-free, `pactl`-free way to target desktop audio; a
        // plain Capture stream (the mic) auto-connects to the default source.
        if matches!(source, Source::DesktopMonitor) {
            props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
        }
        StreamRc::new(core, name, props)
    }

    fn setup_core_listener(core: &mut CoreRc) -> pw::core::Listener {
        core.add_listener_local()
            .info(|i| log::debug!("AUDIO CORE:\n{i:#?}"))
            .error(|e, f, g, h| log::error!("{e},{f},{g},{h}"))
            .done(|d, _| log::debug!("DONE: {d}"))
            .register()
    }

    /// Desktop monitor stream. Reads playback audio and, if a mic buffer is
    /// present, sums the next mic samples into it (clamped) before emitting one
    /// `RawAudioFrame` per processed buffer. Desktop is the clock master, so the
    /// mixed track inherits the desktop timestamps and stays in A/V sync.
    fn setup_desktop_listener(
        stream: &mut StreamRc,
        ready_state: Arc<ReadyState>,
        controls: Arc<CaptureControls>,
        audio_sender: Sender<RawAudioFrame>,
        mic_buffer: Option<MicBuffer>,
    ) -> Result<StreamListener<UserData>, pw::Error> {
        let ready_state_clone = Arc::clone(&ready_state);

        let stream_listener = stream
            .add_local_listener_with_user_data(UserData::default())
            .state_changed(move |_, _, old, new| {
                log::info!("Audio Stream State Changed: {old:?} -> {new:?}");
                ready_state.audio.store(
                    new == StreamState::Streaming,
                    std::sync::atomic::Ordering::Release,
                );
            })
            .param_changed(|_, udata, id, param| {
                let Some(param) = param else {
                    return;
                };
                if id != pw::spa::param::ParamType::Format.as_raw() {
                    return;
                }

                let (media_type, media_subtype) =
                    match pw::spa::param::format_utils::parse_format(param) {
                        Ok(v) => v,
                        Err(_) => return,
                    };

                if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                    return;
                }

                udata
                    .audio_format
                    .parse(param)
                    .expect("Failed to parse audio params");

                log::debug!(
                    "Capturing Rate:{} channels:{}, format: {}",
                    udata.audio_format.rate(),
                    udata.audio_format.channels(),
                    udata.audio_format.format().as_raw()
                );
            })
            .process(move |stream, _| match stream.dequeue_buffer() {
                None => log::debug!("Out of audio buffers"),
                Some(mut buffer) => {
                    if !ready_state_clone.video_ready() || controls.skip_processing() {
                        return;
                    }

                    let datas = buffer.datas_mut();
                    if datas.is_empty() {
                        return;
                    }

                    let data = &mut datas[0];
                    let n_samples = data.chunk().size() / (std::mem::size_of::<f32>()) as u32;

                    if let Some(samples) = data.data() {
                        let samples_f32: &[f32] = bytemuck::cast_slice(samples);
                        let mut mixed = samples_f32[..n_samples as usize].to_vec();

                        // Mix in microphone audio sample-for-sample. Both streams
                        // are forced to 48 kHz stereo, so popping one mic sample
                        // per desktop sample preserves L/R interleaving. Missing
                        // mic samples (underrun) leave the desktop audio as-is.
                        if let Some(buf) = mic_buffer.as_ref() {
                            let mut mic = buf.borrow_mut();
                            for out in mixed.iter_mut() {
                                match mic.pop_front() {
                                    Some(m) => *out = (*out + m).clamp(-1.0, 1.0),
                                    None => break,
                                }
                            }
                        }

                        match audio_sender.try_send(RawAudioFrame {
                            samples: mixed,
                            timestamp: unsafe { pw_stream_get_nsec(stream.as_raw_ptr()) } as i64,
                        }) {
                            Ok(_) => {}
                            Err(crossbeam::channel::TrySendError::Full(frame)) => {
                                log::error!(
                                    "channel is full when trying to send frame at: {}.",
                                    frame.timestamp
                                );
                            }
                            Err(crossbeam::channel::TrySendError::Disconnected(frame)) => {
                                log::error!(
                                    "channel is disconnected when trying to send frame at: {}.",
                                    frame.timestamp
                                );
                            }
                        }
                    }
                }
            })
            .register()?;

        Ok(stream_listener)
    }

    /// Microphone stream. Appends captured samples to the shared mix buffer,
    /// trimming the oldest if it exceeds [`MIC_BUFFER_CAP`] to bound latency.
    fn setup_mic_listener(
        stream: &mut StreamRc,
        mic_buffer: MicBuffer,
    ) -> Result<StreamListener<UserData>, pw::Error> {
        let stream_listener = stream
            .add_local_listener_with_user_data(UserData::default())
            .param_changed(|_, udata, id, param| {
                let Some(param) = param else {
                    return;
                };
                if id != pw::spa::param::ParamType::Format.as_raw() {
                    return;
                }
                let (media_type, media_subtype) =
                    match pw::spa::param::format_utils::parse_format(param) {
                        Ok(v) => v,
                        Err(_) => return,
                    };
                if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                    return;
                }
                udata
                    .audio_format
                    .parse(param)
                    .expect("Failed to parse mic audio params");
            })
            .process(move |stream, _| {
                if let Some(mut buffer) = stream.dequeue_buffer() {
                    let datas = buffer.datas_mut();
                    if datas.is_empty() {
                        return;
                    }
                    let data = &mut datas[0];
                    let n_samples = data.chunk().size() / (std::mem::size_of::<f32>()) as u32;
                    if let Some(samples) = data.data() {
                        let samples_f32: &[f32] = bytemuck::cast_slice(samples);
                        let mut mic = mic_buffer.borrow_mut();
                        mic.extend(&samples_f32[..n_samples as usize]);
                        while mic.len() > MIC_BUFFER_CAP {
                            mic.pop_front();
                        }
                    }
                }
            })
            .register()?;

        Ok(stream_listener)
    }

    fn connect_stream(stream: &mut StreamRc) -> Result<(), pw::Error> {
        // Force 48 kHz stereo F32 so the desktop and mic streams produce
        // sample-aligned PCM (PipeWire inserts a resampler/remix as needed).
        let audio_spa_obj = pw::spa::pod::object! {
            pw::spa::utils::SpaTypes::ObjectParamFormat,
            pw::spa::param::ParamType::EnumFormat,
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::MediaType,
                Id,
                pw::spa::param::format::MediaType::Audio
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::MediaSubtype,
                Id,
                pw::spa::param::format::MediaSubtype::Raw
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::AudioFormat,
                Id,
                pw::spa::param::audio::AudioFormat::F32LE
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::AudioRate,
                Int,
                MIX_RATE as i32
            ),
            pw::spa::pod::property!(
                pw::spa::param::format::FormatProperties::AudioChannels,
                Int,
                MIX_CHANNELS as i32
            )
        };

        let audio_spa_values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &pw::spa::pod::Value::Object(audio_spa_obj),
        )
        .unwrap()
        .0
        .into_inner();

        let mut audio_params = [Pod::from_bytes(&audio_spa_values).unwrap()];

        // No explicit target node: AUTOCONNECT picks the default, and the
        // `stream.capture.sink` property (set on the desktop stream) decides
        // monitor-vs-source.
        stream.connect(
            Direction::Input,
            None,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
            &mut audio_params,
        )
    }

    pub fn run(&mut self) {
        let terminate_loop = self.pipewire_state.pw_loop.clone();
        let terminate_recv = self.termination_recv.take().unwrap();
        let _recv = terminate_recv.attach(self.pipewire_state.pw_loop.loop_(), move |_| {
            log::debug!("Terminating audio capture loop");
            terminate_loop.quit();
        });

        log::debug!("Audio Stream: {:?}", self.pipewire_state._stream);
        self.pipewire_state.pw_loop.run();
    }
}
