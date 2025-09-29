//! Helpers for managing GPU instance buffers used by batched primitives.

use std::marker::PhantomData;

use wgpu::util::{BufferInitDescriptor, DeviceExt, StagingBelt};

/// Manages GPU-side buffers that store per-instance data for batched rendering.
#[derive(Debug)]
pub struct InstanceBufferManager<T: bytemuck::Pod + bytemuck::Zeroable> {
    buffer: Option<wgpu::Buffer>,
    capacity: usize,
    count: usize,
    instances: Vec<T>,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> InstanceBufferManager<T> {
    pub fn new() -> Self {
        Self {
            buffer: None,
            capacity: 0,
            count: 0,
            instances: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Adds an instance to the pending upload list.
    #[inline]
    pub fn add_instance(&mut self, instance: T) {
        self.instances.push(instance);
    }

    /// Uploads accumulated instances to GPU memory, creating or resizing the
    /// backing buffer as needed. Returns the GPU buffer when data is present.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut StagingBelt,
    ) -> Option<&wgpu::Buffer> {
        if self.instances.is_empty() {
            self.count = 0;
            return self.buffer.as_ref();
        }

        let required_instances = self.instances.len();
        let instance_size = std::mem::size_of::<T>();

        if self.capacity < required_instances {
            let new_capacity =
                (required_instances + required_instances / 4).max(64);
            let buffer_size =
                (new_capacity * instance_size) as wgpu::BufferAddress;

            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iced_wgpu::primitive instance buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.capacity = new_capacity;
        }

        if let Some(buffer) = &self.buffer {
            let bytes = bytemuck::cast_slice(&self.instances);

            belt.write_buffer(
                encoder,
                buffer,
                0,
                wgpu::BufferSize::new((bytes.len()) as u64)
                    .expect("non-zero instance data"),
                device,
            )
            .copy_from_slice(bytes);

            self.count = required_instances;
        }

        self.instances.clear();
        self.buffer.as_ref()
    }

    #[inline]
    pub fn instance_count(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn pending_count(&self) -> usize {
        self.instances.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn clear(&mut self) {
        self.instances.clear();
        self.count = 0;
    }

    #[inline]
    pub fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0 && self.instances.is_empty()
    }

    /// Creates a manager with an immediately available GPU buffer containing
    /// the provided instances.
    pub fn create_with_data(device: &wgpu::Device, instances: &[T]) -> Self {
        if instances.is_empty() {
            return Self::new();
        }

        let bytes = bytemuck::cast_slice(instances);
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("iced_wgpu::primitive instance buffer (immediate)"),
            contents: bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            buffer: Some(buffer),
            capacity: instances.len(),
            count: instances.len(),
            instances: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn memory_stats(&self) -> BufferMemoryStats {
        BufferMemoryStats {
            capacity: self.capacity,
            count: self.count,
            pending: self.instances.len(),
            gpu_memory_bytes: self.capacity * std::mem::size_of::<T>(),
            ram_memory_bytes: self.instances.capacity()
                * std::mem::size_of::<T>(),
        }
    }
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> Default
    for InstanceBufferManager<T>
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferMemoryStats {
    pub capacity: usize,
    pub count: usize,
    pub pending: usize,
    pub gpu_memory_bytes: usize,
    pub ram_memory_bytes: usize,
}

impl BufferMemoryStats {
    pub fn total_bytes(&self) -> usize {
        self.gpu_memory_bytes + self.ram_memory_bytes
    }
}
