use std::ffi::c_void;

use cust::sys::{CUcontext, CUstream};

#[cfg(feature = "egl")]
use cust::sys::{CUgraphicsResource, CUresult};
#[cfg(feature = "egl")]
use gl::types::{GLenum, GLuint};
#[cfg(feature = "egl")]
use libc::c_uint;

#[repr(C)]
pub struct AVCUDADeviceContext {
    pub cuda_ctx: CUcontext,
    pub stream: CUstream,
    pub internal: *mut c_void,
}

#[cfg(feature = "egl")]
unsafe extern "C" {
    pub fn cuGraphicsGLRegisterImage(
        resource: *mut CUgraphicsResource,
        image: GLuint,
        target: GLenum,
        flags: c_uint,
    ) -> CUresult;
}
