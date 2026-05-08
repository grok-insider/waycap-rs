use crate::{
    encoders::video::{PipewireSPA, StartVideoEncoder},
    types::video_frame::RawVideoFrame,
    VideoEncoder,
};
#[cfg(feature = "vaapi")]
use crate::VaapiEncoder;
#[cfg(feature = "nvidia")]
use crate::NvencEncoder;
#[cfg(all(feature = "vaapi", feature = "nvidia"))]
use crate::types::error::WaycapError;
#[cfg(all(feature = "vaapi", feature = "nvidia", feature = "vulkan"))]
use crate::waycap_vulkan::{detect_gpu_vendor, GpuVendor};
#[cfg(all(feature = "vaapi", feature = "nvidia", feature = "egl"))]
use crate::waycap_egl::{detect_gpu_vendor, GpuVendor};
use crossbeam::channel::Receiver;

use crate::types::error::Result;

/// "Encoder" which provides the raw DMA-Buf pointers directly.
///
/// Allows for using the image directly on the GPU, which makes it far more performant when, for example, trying to display it to a user.
/// The implementations of [`crate::NvencEncoder`] and [`crate::VaapiEncoder`] show how a [`RawVideoFrame`] can be used.
#[derive(Default)]
pub struct DmaBufEncoder {
    receiver: Option<Receiver<RawVideoFrame>>,
}

impl StartVideoEncoder for DmaBufEncoder {
    fn start_processing(
        capture: &mut crate::Capture<Self>,
        input: Receiver<RawVideoFrame>,
    ) -> Result<()> {
        capture
            .video_encoder
            .as_mut()
            .expect("start_processing should be called after Capture.video_encoder is set")
            .lock()
            .unwrap()
            .receiver = Some(input);
        Ok(())
    }
}
impl VideoEncoder for DmaBufEncoder {
    type Output = RawVideoFrame;

    fn reset(&mut self) -> crate::types::error::Result<()> {
        Ok(())
    }

    fn output(&mut self) -> Option<Receiver<Self::Output>> {
        self.receiver.clone()
    }

    fn drop_processor(&mut self) {}

    fn drain(&mut self) -> crate::types::error::Result<()> {
        Ok(())
    }

    fn get_encoder(&self) -> &Option<ffmpeg_next::codec::encoder::Video> {
        &None
    }
}

impl PipewireSPA for DmaBufEncoder {
    #[allow(unreachable_code)]
    fn get_spa_definition() -> Result<pipewire::spa::pod::Object> {
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
