#[cfg(feature = "nvidia")]
use crate::types::{
    error::Result,
    video_frame::{DmaBufPlane, RawVideoFrame},
};

pub const TIME_UNIT_NS: u64 = 1_000_000_000;

/// Gain applied to the microphone when mixed under the desktop track. Mic sits
/// below the game/voice-chat audio so a loud mic doesn't dominate.
pub const MIC_MIX_GAIN: f32 = 0.7;

/// Soft clipper: transparent below `THRESH`, then a smooth knee that asymptotes
/// to ±1 instead of hard-clipping. Summing desktop + mic (or boosting quiet
/// audio) can push samples past ±1; hard clamping there sounds harsh/saturated,
/// so we round the peaks off gracefully.
pub fn soft_clip(x: f32) -> f32 {
    const THRESH: f32 = 0.8;
    let a = x.abs();
    if a <= THRESH {
        x
    } else {
        let over = a - THRESH;
        let knee = 1.0 - THRESH;
        // over/(over+knee) -> 0..1 as `over` grows, so output -> THRESH..1.0.
        x.signum() * (THRESH + knee * (over / (over + knee)))
    }
}

/// Mix one desktop sample with one (centered) mic sample, with headroom + soft
/// clipping so the combined signal can't hard-clip.
pub fn mix_desktop_mic(desktop: f32, mic: f32) -> f32 {
    soft_clip(desktop + mic * MIC_MIX_GAIN)
}

#[cfg(feature = "nvidia")]
pub fn extract_dmabuf_planes(raw_frame: &RawVideoFrame) -> Result<Vec<DmaBufPlane>> {
    match raw_frame.dmabuf_fd {
        Some(fd) => Ok(vec![DmaBufPlane {
            fd,
            offset: raw_frame.offset,
            stride: raw_frame.stride as u32,
        }]),
        None => Err("No DMA-BUF file descriptor in frame".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_clip_passes_in_range() {
        for &x in &[0.0, 0.1, -0.3, 0.5, -0.79, 0.8, -0.8] {
            assert!((soft_clip(x) - x).abs() < 1e-6, "{x} changed");
        }
    }

    #[test]
    fn soft_clip_bounds_overshoot() {
        for &x in &[1.0, 1.5, 2.0, 5.0, -1.0, -3.0, -10.0] {
            let y = soft_clip(x);
            assert!(y.abs() < 1.0, "soft_clip({x}) = {y} not < 1");
            assert_eq!(y.signum(), x.signum());
        }
        // Monotonic + compressing: bigger input -> bigger (but bounded) output.
        assert!(soft_clip(2.0) > soft_clip(1.2));
    }

    #[test]
    fn mix_never_hard_clips() {
        // Two full-scale sources must not produce a hard-clipped ±1 sample.
        for &(d, m) in &[(1.0, 1.0), (-1.0, -1.0), (0.9, 0.9), (1.0, -1.0)] {
            let y = mix_desktop_mic(d, m);
            assert!(y.abs() < 1.0, "mix({d},{m}) = {y} clipped");
        }
        // Quiet desktop with no mic is untouched.
        assert!((mix_desktop_mic(0.2, 0.0) - 0.2).abs() < 1e-6);
    }
}
