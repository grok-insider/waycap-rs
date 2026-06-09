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
#[derive(Debug, Clone, Copy, Default)]
pub enum RateControl {
    /// Constant-quality VBR derived from the [`QualityPreset`] (the default).
    #[default]
    Quality,
    /// Constant bitrate at the given kbit/s. Keeps the encoded output rate
    /// predictable in high-motion scenes — what a RAM replay buffer wants.
    ConstantBitrate { kbps: u32 },
}
