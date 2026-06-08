use crate::types::error::Result;

use ash::vk;
use ash::{Entry, Instance};

#[cfg(feature = "nvidia")]
use crate::types::video_frame::DmaBufPlane;
#[cfg(feature = "nvidia")]
use ash::{ext, khr, Device};
#[cfg(feature = "nvidia")]
use std::{ffi::c_void, os::unix::io::RawFd};

#[cfg(feature = "nvidia")]
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

#[cfg(feature = "vaapi")]
#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum GpuVendor {
    NVIDIA,
    AMD,
    INTEL,
    UNKNOWN,
}

#[cfg(feature = "vaapi")]
impl GpuVendor {
    fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10DE => GpuVendor::NVIDIA,
            0x1002 => GpuVendor::AMD,
            0x8086 => GpuVendor::INTEL,
            _ => {
                log::error!("Unknown GPU vendor ID: 0x{id:04X}");
                GpuVendor::UNKNOWN
            }
        }
    }
}

#[cfg(feature = "vaapi")]
pub fn detect_gpu_vendor() -> Result<GpuVendor> {
    let entry = unsafe { Entry::load() }.map_err(|e| format!("Failed to load Vulkan: {e}"))?;

    let app_name = c"waycap-rs";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .api_version(vk::API_VERSION_1_1);
    let instance_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_ci, None) }
        .map_err(|e| format!("Failed to create Vulkan instance: {e}"))?;

    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("Failed to enumerate physical devices: {e}"))?;

    let vendor = physical_devices
        .first()
        .map(|&pd| {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            GpuVendor::from_vendor_id(props.vendor_id)
        })
        .unwrap_or(GpuVendor::UNKNOWN);

    unsafe { instance.destroy_instance(None) };
    Ok(vendor)
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

pub struct VulkanContext {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    #[allow(dead_code)]
    queue_family_index: u32,
    command_pool: vk::CommandPool,

    external_memory_fd: khr::external_memory_fd::Device,

    persistent_buffer: vk::Buffer,
    persistent_buffer_memory: vk::DeviceMemory,
    persistent_buffer_size: u64,

    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
}

impl VulkanContext {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let entry = unsafe { Entry::load() }.map_err(|e| format!("Failed to load Vulkan: {e}"))?;

        let app_name = c"waycap-rs";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .api_version(vk::API_VERSION_1_1);

        let instance_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(|e| format!("Failed to create Vulkan instance: {e}"))?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("Failed to enumerate physical devices: {e}"))?;
        if physical_devices.is_empty() {
            return Err("No Vulkan physical devices found".into());
        }

        let (physical_device, queue_family_index) = physical_devices
            .iter()
            .find_map(|&pd| {
                let qfs = unsafe { instance.get_physical_device_queue_family_properties(pd) };
                qfs.iter().enumerate().find_map(|(i, qf)| {
                    if qf
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER)
                    {
                        Some((pd, i as u32))
                    } else {
                        None
                    }
                })
            })
            .ok_or("No suitable Vulkan queue family found")?;

        let queue_priorities = [1.0_f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_exts = [
            khr::external_memory_fd::NAME.as_ptr(),
            ext::external_memory_dma_buf::NAME.as_ptr(),
            ext::image_drm_format_modifier::NAME.as_ptr(),
        ];
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&device_exts);
        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(|e| format!("Failed to create Vulkan device: {e}"))?;

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| format!("Failed to create command pool: {e}"))?;

        let external_memory_fd = khr::external_memory_fd::Device::new(&instance, &device);

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let persistent_buffer_size = (width * height * 4) as u64;
        let (persistent_buffer, persistent_buffer_memory) =
            Self::create_persistent_buffer(&device, &mem_props, persistent_buffer_size)?;

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            external_memory_fd,
            persistent_buffer,
            persistent_buffer_memory,
            persistent_buffer_size,
            width,
            height,
        })
    }

    fn create_persistent_buffer(
        device: &Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let mut export_info = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let buf_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut export_info);

        let buffer = unsafe { device.create_buffer(&buf_ci, None) }
            .map_err(|e| format!("Failed to create persistent buffer: {e}"))?;

        let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_type_index = find_memory_type(
            mem_props,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or("No DEVICE_LOCAL memory type for persistent buffer")?;

        let mut export_alloc = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_alloc);

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|e| {
            unsafe { device.destroy_buffer(buffer, None) };
            format!("Failed to allocate persistent buffer memory: {e}")
        })?;

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.map_err(|e| {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            format!("Failed to bind persistent buffer memory: {e}")
        })?;

        Ok((buffer, memory))
    }

    pub fn export_persistent_memory_fd(&self) -> Result<RawFd> {
        let get_fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(self.persistent_buffer_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        unsafe { self.external_memory_fd.get_memory_fd(&get_fd_info) }
            .map_err(|e| format!("Failed to export Vulkan memory FD: {e}").into())
    }

    pub fn get_persistent_buffer_size(&self) -> u64 {
        self.persistent_buffer_size
    }

    pub fn copy_dmabuf_to_persistent_buffer(
        &self,
        planes: &[DmaBufPlane],
        modifier: u64,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let plane = planes.first().ok_or("No DMA-BUF planes provided")?;

        let dup_fd = unsafe { libc::dup(plane.fd) };
        if dup_fd < 0 {
            return Err("Failed to dup DMA-BUF fd".into());
        }

        let (temp_image, temp_memory) = self
            .import_dmabuf_as_image(dup_fd, plane, modifier, width, height)
            .inspect_err(|_| {
                unsafe { libc::close(dup_fd) };
            })?;

        let copy_result = self.record_and_submit_copy(temp_image, width, height);

        unsafe {
            self.device.destroy_image(temp_image, None);
            self.device.free_memory(temp_memory, None);
        }

        copy_result
    }

    fn import_dmabuf_as_image(
        &self,
        fd: RawFd,
        plane: &DmaBufPlane,
        modifier: u64,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        let plane_layout = vk::SubresourceLayout {
            offset: plane.offset as u64,
            size: 0,
            row_pitch: plane.stride as u64,
            array_pitch: 0,
            depth_pitch: 0,
        };

        let effective_modifier = if modifier == DRM_FORMAT_MOD_INVALID {
            0
        } else {
            modifier
        };

        let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(effective_modifier)
            .plane_layouts(std::slice::from_ref(&plane_layout));

        let mut external_image_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        external_image_info.p_next = (&raw mut modifier_info).cast::<c_void>();

        let image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_image_info);

        let image = unsafe { self.device.create_image(&image_ci, None) }
            .map_err(|e| format!("Failed to create DMA-BUF image: {e}"))?;

        let mut dedicated_reqs = vk::MemoryDedicatedRequirements::default();
        let mut mem_reqs2 = vk::MemoryRequirements2 {
            p_next: (&raw mut dedicated_reqs).cast::<c_void>(),
            ..Default::default()
        };

        let image_mem_reqs_info = vk::ImageMemoryRequirementsInfo2::default().image(image);
        unsafe {
            self.device
                .get_image_memory_requirements2(&image_mem_reqs_info, &mut mem_reqs2)
        };

        let memory_type_index = find_memory_type(
            &mem_props,
            mem_reqs2.memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or("No suitable memory type for DMA-BUF import")?;

        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd);

        let mut dedicated_alloc = vk::MemoryDedicatedAllocateInfo::default().image(image);
        dedicated_alloc.p_next = (&raw mut import_info).cast::<c_void>();

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs2.memory_requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut dedicated_alloc);

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }.map_err(|e| {
            unsafe { self.device.destroy_image(image, None) };
            format!("Failed to allocate DMA-BUF memory: {e}")
        })?;

        unsafe { self.device.bind_image_memory(image, memory, 0) }.map_err(|e| {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_image(image, None);
            }
            format!("Failed to bind DMA-BUF image memory: {e}")
        })?;

        Ok((image, memory))
    }

    fn record_and_submit_copy(&self, src_image: vk::Image, width: u32, height: u32) -> Result<()> {
        let cmd_buf = alloc_cmd_buf(&self.device, self.command_pool)?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd_buf, &begin_info) }
            .map_err(|e| format!("begin_command_buffer: {e}"))?;

        let src_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(src_image)
            .subresource_range(color_subresource_range())
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&src_barrier),
            );
        }

        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        unsafe {
            self.device.cmd_copy_image_to_buffer(
                cmd_buf,
                src_image,
                vk::ImageLayout::GENERAL,
                self.persistent_buffer,
                std::slice::from_ref(&region),
            );
        }

        let buf_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.persistent_buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&buf_barrier),
                &[],
            );
        }

        submit_and_wait(&self.device, self.queue, cmd_buf)?;
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &[cmd_buf])
        };

        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_buffer(self.persistent_buffer, None);
            self.device.free_memory(self.persistent_buffer_memory, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(flags)
    })
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn alloc_cmd_buf(device: &Device, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let bufs = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| format!("Failed to allocate command buffer: {e}"))?;
    Ok(bufs[0])
}

fn submit_and_wait(device: &Device, queue: vk::Queue, cmd_buf: vk::CommandBuffer) -> Result<()> {
    unsafe { device.end_command_buffer(cmd_buf) }
        .map_err(|e| format!("end_command_buffer: {e}"))?;

    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .map_err(|e| format!("create_fence: {e}"))?;

    let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));
    unsafe {
        device
            .queue_submit(queue, std::slice::from_ref(&submit), fence)
            .map_err(|e| format!("queue_submit: {e}"))?;
        device
            .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
            .map_err(|e| format!("wait_for_fences: {e}"))?;
        device.destroy_fence(fence, None);
    }
    Ok(())
}
