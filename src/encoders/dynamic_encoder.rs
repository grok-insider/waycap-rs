use crossbeam::channel::Receiver;
use ffmpeg_next::codec::encoder;

use crate::{
    encoders::video::{PipewireSPA, ProcessingThread},
    types::{
        config::VideoEncoder as VideoEncoderType,
        error::Result,
        video_frame::{EncodedVideoFrame, RawVideoFrame},
    },
    VideoEncoder,
};

#[cfg(feature = "vaapi")]
use crate::encoders::vaapi_encoder::VaapiEncoder;

#[cfg(feature = "nvidia")]
use crate::encoders::nvenc_encoder::NvencEncoder;
#[cfg(all(feature = "vaapi", feature = "nvidia"))]
use crate::types::error::WaycapError;

// detect_gpu_vendor is only needed when both encoders are available for auto-selection
#[cfg(all(feature = "vaapi", feature = "nvidia", feature = "egl"))]
use crate::waycap_egl::{detect_gpu_vendor, GpuVendor};
#[cfg(all(feature = "vaapi", feature = "nvidia", feature = "vulkan"))]
use crate::waycap_vulkan::{detect_gpu_vendor, GpuVendor};

pub enum DynamicEncoder {
    #[cfg(feature = "vaapi")]
    Vaapi(VaapiEncoder),
    #[cfg(feature = "nvidia")]
    Nvenc(NvencEncoder),
}

impl DynamicEncoder {
    pub(crate) fn new(
        encoder_type: Option<VideoEncoderType>,
        width: u32,
        height: u32,
        quality_preset: crate::types::config::QualityPreset,
    ) -> crate::types::error::Result<DynamicEncoder> {
        let encoder_type = match encoder_type {
            Some(typ) => typ,
            // Both available: detect GPU to pick
            #[cfg(all(feature = "vaapi", feature = "nvidia"))]
            None => match detect_gpu_vendor()? {
                GpuVendor::NVIDIA => VideoEncoderType::H264Nvenc,
                GpuVendor::AMD | GpuVendor::INTEL => VideoEncoderType::H264Vaapi,
                GpuVendor::UNKNOWN => {
                    return Err(WaycapError::Init(
                        "Unknown/Unimplemented GPU vendor".to_string(),
                    ));
                }
            },
            // Only one available: use it directly
            #[cfg(all(feature = "vaapi", not(feature = "nvidia")))]
            None => VideoEncoderType::H264Vaapi,
            #[cfg(all(feature = "nvidia", not(feature = "vaapi")))]
            None => VideoEncoderType::H264Nvenc,
        };
        Ok(match encoder_type {
            #[cfg(feature = "vaapi")]
            VideoEncoderType::H264Vaapi => {
                DynamicEncoder::Vaapi(VaapiEncoder::new(width, height, quality_preset)?)
            }
            #[cfg(feature = "nvidia")]
            VideoEncoderType::H264Nvenc => {
                DynamicEncoder::Nvenc(NvencEncoder::new(width, height, quality_preset)?)
            }
        })
    }
}

impl VideoEncoder for DynamicEncoder {
    type Output = EncodedVideoFrame;

    fn reset(&mut self) -> Result<()> {
        match self {
            #[cfg(feature = "vaapi")]
            DynamicEncoder::Vaapi(enc) => enc.reset(),
            #[cfg(feature = "nvidia")]
            DynamicEncoder::Nvenc(enc) => enc.reset(),
        }
    }

    fn output(&mut self) -> Option<Receiver<Self::Output>> {
        match self {
            #[cfg(feature = "vaapi")]
            DynamicEncoder::Vaapi(enc) => enc.output(),
            #[cfg(feature = "nvidia")]
            DynamicEncoder::Nvenc(enc) => enc.output(),
        }
    }

    fn drop_processor(&mut self) {
        match self {
            #[cfg(feature = "vaapi")]
            DynamicEncoder::Vaapi(enc) => enc.drop_processor(),
            #[cfg(feature = "nvidia")]
            DynamicEncoder::Nvenc(enc) => enc.drop_processor(),
        }
    }

    fn drain(&mut self) -> Result<()> {
        match self {
            #[cfg(feature = "vaapi")]
            DynamicEncoder::Vaapi(enc) => enc.drain(),
            #[cfg(feature = "nvidia")]
            DynamicEncoder::Nvenc(enc) => enc.drain(),
        }
    }

    fn get_encoder(&self) -> &Option<encoder::Video> {
        match self {
            #[cfg(feature = "vaapi")]
            DynamicEncoder::Vaapi(enc) => enc.get_encoder(),
            #[cfg(feature = "nvidia")]
            DynamicEncoder::Nvenc(enc) => enc.get_encoder(),
        }
    }
}

impl ProcessingThread for DynamicEncoder {
    fn process(&mut self, frame: RawVideoFrame) -> Result<()> {
        match self {
            #[cfg(feature = "vaapi")]
            DynamicEncoder::Vaapi(enc) => enc.process(frame),
            #[cfg(feature = "nvidia")]
            DynamicEncoder::Nvenc(enc) => enc.process(frame),
        }
    }

    fn thread_setup(&mut self) -> Result<()> {
        match self {
            #[cfg(feature = "vaapi")]
            DynamicEncoder::Vaapi(enc) => enc.thread_setup(),
            #[cfg(feature = "nvidia")]
            DynamicEncoder::Nvenc(enc) => enc.thread_setup(),
        }
    }

    fn thread_teardown(&mut self) -> Result<()> {
        match self {
            #[cfg(feature = "vaapi")]
            DynamicEncoder::Vaapi(enc) => enc.thread_teardown(),
            #[cfg(feature = "nvidia")]
            DynamicEncoder::Nvenc(enc) => enc.thread_teardown(),
        }
    }
}

impl PipewireSPA for DynamicEncoder {
    #[allow(unreachable_code)]
    fn get_spa_definition() -> Result<pipewire::spa::pod::Object> {
        // Both available: detect GPU to pick the right format
        #[cfg(all(feature = "vaapi", feature = "nvidia"))]
        return match detect_gpu_vendor()? {
            GpuVendor::NVIDIA => NvencEncoder::get_spa_definition(),
            GpuVendor::AMD | GpuVendor::INTEL => VaapiEncoder::get_spa_definition(),
            GpuVendor::UNKNOWN => Err(WaycapError::Init(
                "Unknown/Unimplemented GPU vendor".to_string(),
            )),
        };
        #[cfg(all(feature = "vaapi", not(feature = "nvidia")))]
        return VaapiEncoder::get_spa_definition();
        #[cfg(all(feature = "nvidia", not(feature = "vaapi")))]
        return NvencEncoder::get_spa_definition();
        unreachable!()
    }
}
