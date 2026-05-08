#[derive(Debug, Clone, Copy)]
pub enum VideoEncoder {
    #[cfg(feature = "nvidia")]
    H264Nvenc,
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
