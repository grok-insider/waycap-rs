use crossbeam::channel::Receiver;

use crate::types::{
    audio_frame::{EncodedAudioFrame, RawAudioFrame},
    error::Result,
};

pub trait AudioEncoder: Send {
    fn new() -> Result<Self>
    where
        Self: Sized;
    fn process(&mut self, raw_frame: RawAudioFrame) -> Result<()>;
    fn drain(&mut self) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
    fn get_encoder(&self) -> &Option<ffmpeg_next::codec::encoder::Audio>;
    fn get_encoded_recv(&mut self) -> Option<Receiver<EncodedAudioFrame>>;
    fn drop_encoder(&mut self);
}
