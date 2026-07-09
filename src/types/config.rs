#[derive(Debug, Clone, Copy)]
pub enum VideoEncoder {
    #[cfg(feature = "nvidia")]
    H264Nvenc,
    #[cfg(feature = "nvidia")]
    HevcNvenc,
    #[cfg(feature = "nvidia")]
    Av1Nvenc,
    #[cfg(feature = "vaapi")]
    H264Vaapi,
}

#[derive(Debug, Clone, Copy)]
pub enum AudioEncoder {
    Opus,
}

#[derive(Debug, Clone, Copy)]
pub enum QualityPreset {
    Low,
    Medium,
    High,
    Ultra,
}

/// Encoder rate control.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RateControl {
    /// Constant-quality VBR derived from the [`QualityPreset`] (the default).
    #[default]
    Quality,
    /// Constant bitrate at the given kbit/s. Keeps the encoded output rate
    /// predictable in high-motion scenes — what a RAM replay buffer wants.
    ConstantBitrate { kbps: u32 },
}

/// Default GOP length in frames (matches the historical hardcode).
pub const DEFAULT_GOP_SIZE: u32 = 30;

/// NVENC (and similar) rate-distortion / latency tune.
///
/// Default is [`EncoderTune::Quality`] (`hq`), matching the previous hardcode
/// so callers that never set a tune keep the old behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EncoderTune {
    /// High quality (`tune=hq`).
    #[default]
    Quality,
    /// Low latency (`tune=ll`).
    Performance,
}

impl EncoderTune {
    /// Value for ffmpeg's `tune` option.
    pub fn as_ffmpeg_tune(self) -> &'static str {
        match self {
            EncoderTune::Quality => "hq",
            EncoderTune::Performance => "ll",
        }
    }
}

/// Encoded luma/chroma sample value range.
///
/// Default is [`ColorRange::Limited`] (TV / MPEG range).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColorRange {
    #[default]
    Limited,
    Full,
}

impl ColorRange {
    /// ffmpeg `color_range` dictionary value (`tv` / `pc`).
    pub fn as_ffmpeg_color_range(self) -> &'static str {
        match self {
            ColorRange::Limited => "tv",
            ColorRange::Full => "pc",
        }
    }
}

/// Video encode knobs that used to be hard-coded on the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Frames between IDR/keyframes (GOP length).
    pub gop_size: u32,
    pub tune: EncoderTune,
    pub color_range: ColorRange,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            gop_size: DEFAULT_GOP_SIZE,
            tune: EncoderTune::Quality,
            color_range: ColorRange::Limited,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_legacy_hardcodes() {
        let o = EncodeOptions::default();
        assert_eq!(o.gop_size, 30);
        assert_eq!(o.tune.as_ffmpeg_tune(), "hq");
        assert_eq!(o.color_range.as_ffmpeg_color_range(), "tv");
    }

    #[test]
    fn performance_tune_is_low_latency() {
        assert_eq!(EncoderTune::Performance.as_ffmpeg_tune(), "ll");
    }

    #[test]
    fn full_range_maps_to_pc() {
        assert_eq!(ColorRange::Full.as_ffmpeg_color_range(), "pc");
    }
}
