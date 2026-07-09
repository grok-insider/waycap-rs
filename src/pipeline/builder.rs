use crate::{
    encoders::dynamic_encoder::DynamicEncoder,
    types::{
        config::{
            AudioEncoder, ColorRange, EncodeOptions, EncoderTune, QualityPreset, RateControl,
            VideoEncoder, DEFAULT_GOP_SIZE,
        },
        error::Result,
    },
    Capture,
};

pub struct CaptureBuilder {
    video_encoder: Option<VideoEncoder>,
    audio_encoder: Option<AudioEncoder>,
    quality_preset: Option<QualityPreset>,
    rate_control: RateControl,
    encode: EncodeOptions,
    include_cursor: bool,
    include_audio: bool,
    include_mic: bool,
    target_fps: u64,
    restore_token: Option<String>,
}

impl Default for CaptureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureBuilder {
    pub fn new() -> Self {
        Self {
            video_encoder: None,
            audio_encoder: None,
            quality_preset: None,
            rate_control: RateControl::default(),
            encode: EncodeOptions::default(),
            include_cursor: false,
            include_audio: false,
            include_mic: false,
            target_fps: 60,
            restore_token: None,
        }
    }

    /// Optional: Force use a specific video encoder.
    /// Default: Uses EGL to determine GPU at runtime.
    pub fn with_video_encoder(mut self, encoder: VideoEncoder) -> Self {
        self.video_encoder = Some(encoder);
        self
    }

    /// Optional: Force use a specific audio encoder.
    /// Default: Opus audio encoder.
    pub fn with_audio_encoder(mut self, encoder: AudioEncoder) -> Self {
        self.audio_encoder = Some(encoder);
        self
    }

    pub fn with_cursor_shown(mut self) -> Self {
        self.include_cursor = true;
        self
    }

    pub fn with_audio(mut self) -> Self {
        self.include_audio = true;
        self
    }

    /// Optional: also capture the default microphone and mix it into the single
    /// audio track. Implies [`Self::with_audio`]. The mix happens on the same
    /// PipeWire clock as the desktop monitor, so it stays in A/V sync.
    pub fn with_microphone(mut self) -> Self {
        self.include_audio = true;
        self.include_mic = true;
        self
    }

    pub fn with_quality_preset(mut self, quality: QualityPreset) -> Self {
        self.quality_preset = Some(quality);
        self
    }

    /// Optional: encoder rate control. Default: constant-quality VBR from the
    /// quality preset. Use [`RateControl::ConstantBitrate`] for a predictable
    /// output rate (e.g. a RAM replay buffer in high-motion scenes).
    pub fn with_rate_control(mut self, rate_control: RateControl) -> Self {
        self.rate_control = rate_control;
        self
    }

    /// Optional: Set a target FPS for the recording.
    /// Default: 60fps
    pub fn with_target_fps(mut self, fps: u64) -> Self {
        self.target_fps = fps;
        self
    }

    /// Keyframe interval in frames (GOP length). Default: [`DEFAULT_GOP_SIZE`] (30).
    /// Values below 1 are clamped to 1.
    pub fn with_gop(mut self, gop_size: u32) -> Self {
        self.encode.gop_size = gop_size.max(1);
        self
    }

    /// NVENC tune bias. Default: [`EncoderTune::Quality`] (`hq`).
    pub fn with_encoder_tune(mut self, tune: EncoderTune) -> Self {
        self.encode.tune = tune;
        self
    }

    /// Encoded sample value range. Default: [`ColorRange::Limited`].
    pub fn with_color_range(mut self, range: ColorRange) -> Self {
        self.encode.color_range = range;
        self
    }

    /// Optional: Provide a restore token from a previous session to skip the
    /// screen-recording permission prompt. Retrieve the token after a successful
    /// build via [`crate::Capture::restore_token`].
    pub fn with_restore_token(mut self, token: String) -> Self {
        self.restore_token = Some(token);
        self
    }

    pub fn build(self) -> Result<Capture<DynamicEncoder>> {
        let quality = match self.quality_preset {
            Some(qual) => qual,
            None => QualityPreset::Medium,
        };

        let audio_encoder = if self.include_audio {
            match self.audio_encoder {
                Some(enc) => enc,
                None => AudioEncoder::Opus,
            }
        } else {
            AudioEncoder::Opus
        };

        Capture::new(
            self.video_encoder,
            audio_encoder,
            quality,
            self.rate_control,
            self.encode,
            self.include_cursor,
            self.include_audio,
            self.include_mic,
            self.target_fps,
            self.restore_token,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_use_legacy_encode_options() {
        let b = CaptureBuilder::new();
        assert_eq!(b.encode.gop_size, DEFAULT_GOP_SIZE);
        assert_eq!(b.encode.tune, EncoderTune::Quality);
        assert_eq!(b.encode.color_range, ColorRange::Limited);
    }

    #[test]
    fn builder_setters_override_encode_options() {
        let b = CaptureBuilder::new()
            .with_gop(120)
            .with_encoder_tune(EncoderTune::Performance)
            .with_color_range(ColorRange::Full);
        assert_eq!(b.encode.gop_size, 120);
        assert_eq!(b.encode.tune, EncoderTune::Performance);
        assert_eq!(b.encode.color_range, ColorRange::Full);
    }

    #[test]
    fn with_gop_clamps_zero_to_one() {
        let b = CaptureBuilder::new().with_gop(0);
        assert_eq!(b.encode.gop_size, 1);
    }
}
