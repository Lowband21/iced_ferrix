use crate::core::{self, Size};
use crate::graphics::Shell;
use crate::image::atlas::{self, Atlas};

#[cfg(all(feature = "image", not(target_arch = "wasm32")))]
use worker::Worker;

#[cfg(feature = "image")]
use std::collections::HashMap;

use std::sync::Arc;

pub struct Cache {
    atlas: Atlas,
    #[cfg(feature = "image")]
    raster: Raster,
    #[cfg(feature = "svg")]
    vector: crate::image::vector::Cache,
    #[cfg(all(feature = "image", not(target_arch = "wasm32")))]
    worker: Worker,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasRegion {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub layer: u32,
}

impl Cache {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        backend: wgpu::Backend,
        layout: Arc<wgpu::BindGroupLayout>,
        _shell: &Shell,
    ) -> Self {
        #[cfg(all(feature = "image", not(target_arch = "wasm32")))]
        let worker =
            Worker::new(device, _queue, backend, layout.clone(), _shell);

        Self {
            atlas: Atlas::new(device, backend, layout),
            #[cfg(feature = "image")]
            raster: Raster {
                cache: crate::image::raster::Cache::default(),
                pending: HashMap::new(),
                belt: wgpu::util::StagingBelt::new(2 * 1024 * 1024),
            },
            #[cfg(feature = "svg")]
            vector: crate::image::vector::Cache::default(),
            #[cfg(all(feature = "image", not(target_arch = "wasm32")))]
            worker,
        }
    }

    #[cfg(feature = "image")]
    pub fn allocate_image(
        &mut self,
        handle: &core::image::Handle,
        callback: impl FnOnce(Result<core::image::Allocation, core::image::Error>)
        + Send
        + 'static,
    ) {
        use crate::image::raster::Memory;

        let callback = Box::new(callback);

        if let Some(callbacks) = self.raster.pending.get_mut(&handle.id()) {
            callbacks.push(callback);
            return;
        }

        if let Some(Memory::Device {
            allocation, entry, ..
        }) = self.raster.cache.get_mut(handle)
        {
            if let Some(allocation) = allocation
                .as_ref()
                .and_then(core::image::Allocation::upgrade)
            {
                callback(Ok(allocation));
                return;
            }

            #[allow(unsafe_code)]
            let new = unsafe { core::image::allocate(handle, entry.size()) };
            *allocation = Some(new.downgrade());
            callback(Ok(new));

            return;
        }

        let _ = self.raster.pending.insert(handle.id(), vec![callback]);

        #[cfg(not(target_arch = "wasm32"))]
        self.worker.load(handle);
    }

    pub fn texture_layout(&self) -> Arc<wgpu::BindGroupLayout> {
        Arc::clone(self.atlas.bind_group_layout())
    }

    pub fn bind_group(&self) -> &Arc<wgpu::BindGroup> {
        self.atlas.bind_group()
    }

    #[cfg(feature = "image")]
    pub fn load_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: &core::image::Handle,
    ) -> Result<core::image::Allocation, core::image::Error> {
        use crate::image::raster::Memory;

        if !self.raster.cache.contains(handle) {
            self.raster.cache.insert(handle, Memory::load(handle));
        }

        match self.raster.cache.get_mut(handle).unwrap() {
            Memory::Host(image) => {
                let mut encoder = device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("raster image upload"),
                    },
                );

                let entry = self.atlas.upload(
                    device,
                    &mut encoder,
                    &mut self.raster.belt,
                    image.width(),
                    image.height(),
                    image,
                );

                self.raster.belt.finish();
                let submission = queue.submit([encoder.finish()]);
                self.raster.belt.recall();

                let Some(entry) = entry else {
                    return Err(core::image::Error::OutOfMemory);
                };

                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                });

                #[allow(unsafe_code)]
                let allocation = unsafe {
                    core::image::allocate(
                        handle,
                        Size::new(image.width(), image.height()),
                    )
                };

                self.raster.cache.insert(
                    handle,
                    Memory::Device {
                        entry,
                        bind_group: None,
                        allocation: Some(allocation.downgrade()),
                    },
                );

                Ok(allocation)
            }
            Memory::Device {
                entry, allocation, ..
            } => {
                if let Some(allocation) = allocation
                    .as_ref()
                    .and_then(core::image::Allocation::upgrade)
                {
                    return Ok(allocation);
                }

                #[allow(unsafe_code)]
                let new =
                    unsafe { core::image::allocate(handle, entry.size()) };

                *allocation = Some(new.downgrade());

                Ok(new)
            }
            Memory::Error(error) => Err(error.clone()),
        }
    }

    #[cfg(feature = "image")]
    pub fn measure_image(
        &mut self,
        handle: &core::image::Handle,
    ) -> Option<Size<u32>> {
        self.receive();

        let image = load_image(
            &mut self.raster.cache,
            &mut self.raster.pending,
            #[cfg(not(target_arch = "wasm32"))]
            &self.worker,
            handle,
            None,
        )?;

        Some(image.dimensions())
    }

    #[cfg(feature = "svg")]
    pub fn measure_svg(&mut self, handle: &core::svg::Handle) -> Size<u32> {
        // TODO: Concurrency
        self.vector.load(handle).viewport_dimensions()
    }

    #[cfg(feature = "image")]
    pub fn upload_raster(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        handle: &core::image::Handle,
    ) -> Option<(&atlas::Entry, &Arc<wgpu::BindGroup>)> {
        use crate::image::raster::Memory;

        self.receive();

        let memory = load_image(
            &mut self.raster.cache,
            &mut self.raster.pending,
            #[cfg(not(target_arch = "wasm32"))]
            &self.worker,
            handle,
            None,
        )?;

        if let Memory::Device {
            entry, bind_group, ..
        } = memory
        {
            return Some((
                entry,
                bind_group.as_ref().unwrap_or(self.atlas.bind_group()),
            ));
        }

        let image = memory.host()?;

        const MAX_SYNC_SIZE: usize = 2 * 1024 * 1024;

        // TODO: Concurrent Wasm support
        if image.len() < MAX_SYNC_SIZE || cfg!(target_arch = "wasm32") {
            let entry = self.atlas.upload(
                device,
                encoder,
                belt,
                image.width(),
                image.height(),
                &image,
            )?;

            *memory = Memory::Device {
                entry,
                bind_group: None,
                allocation: None,
            };

            if let Memory::Device { entry, .. } = memory {
                return Some((entry, self.atlas.bind_group()));
            }
        }

        if !self.raster.pending.contains_key(&handle.id()) {
            let _ = self.raster.pending.insert(handle.id(), Vec::new());

            #[cfg(not(target_arch = "wasm32"))]
            self.worker.upload(handle, image);
        }

        None
    }

    #[cfg(feature = "image")]
    pub fn ensure_raster_region(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        handle: &core::image::Handle,
    ) -> Option<AtlasRegion> {
        let (entry, _bind_group) =
            self.upload_raster(device, encoder, belt, handle)?;
        Some(Self::region_from_entry(entry))
    }

    #[cfg(feature = "image")]
    pub fn cached_raster_region(
        &mut self,
        handle: &core::image::Handle,
    ) -> Option<AtlasRegion> {
        self.receive();

        // Record cache hits so atlas entries persist across trims.
        self.raster
            .cache
            .atlas_entry(handle)
            .map(Self::region_from_entry)
    }

    #[cfg(feature = "svg")]
    pub fn upload_vector(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        handle: &core::svg::Handle,
        color: Option<core::Color>,
        size: Size,
        scale: f32,
    ) -> Option<(&atlas::Entry, &Arc<wgpu::BindGroup>)> {
        // TODO: Concurrency
        self.vector
            .upload(
                device,
                encoder,
                belt,
                handle,
                color,
                size,
                scale,
                &mut self.atlas,
            )
            .map(|entry| (entry, self.atlas.bind_group()))
    }

    pub fn trim(&mut self) {
        #[cfg(feature = "image")]
        {
            self.receive();
            self.raster.cache.trim(&mut self.atlas, |_bind_group| {
                #[cfg(not(target_arch = "wasm32"))]
                self.worker.drop(_bind_group);
            });
        }

        #[cfg(feature = "svg")]
        self.vector.trim(&mut self.atlas); // TODO: Concurrency
    }

    #[cfg(feature = "image")]
    fn receive(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        while let Ok(work) = self.worker.try_recv() {
            use crate::image::raster::Memory;

            match work {
                worker::Work::Upload {
                    handle,
                    entry,
                    bind_group,
                } => {
                    let callbacks = self.raster.pending.remove(&handle.id());

                    let allocation = if let Some(callbacks) = callbacks {
                        #[allow(unsafe_code)]
                        let allocation = unsafe {
                            core::image::allocate(&handle, entry.size())
                        };

                        let reference = allocation.downgrade();

                        for callback in callbacks {
                            callback(Ok(allocation.clone()));
                        }

                        Some(reference)
                    } else {
                        None
                    };

                    self.raster.cache.insert(
                        &handle,
                        Memory::Device {
                            entry,
                            bind_group: Some(bind_group),
                            allocation,
                        },
                    );
                }
                worker::Work::Error { handle, error } => {
                    let callbacks = self.raster.pending.remove(&handle.id());

                    if let Some(callbacks) = callbacks {
                        for callback in callbacks {
                            callback(Err(error.clone()));
                        }
                    }

                    self.raster.cache.insert(&handle, Memory::Error(error));
                }
            }
        }
    }
}

#[cfg(all(feature = "image", not(target_arch = "wasm32")))]
impl Drop for Cache {
    fn drop(&mut self) {
        self.worker.quit();
    }
}

#[cfg(feature = "image")]
struct Raster {
    cache: crate::image::raster::Cache,
    pending: HashMap<core::image::Id, Vec<Callback>>,
    belt: wgpu::util::StagingBelt,
}

#[cfg(feature = "image")]
type Callback =
    Box<dyn FnOnce(Result<core::image::Allocation, core::image::Error>) + Send>;

#[cfg(feature = "image")]
fn load_image<'a>(
    cache: &'a mut crate::image::raster::Cache,
    pending: &mut HashMap<core::image::Id, Vec<Callback>>,
    #[cfg(not(target_arch = "wasm32"))] worker: &Worker,
    handle: &core::image::Handle,
    callback: Option<Callback>,
) -> Option<&'a mut crate::image::raster::Memory> {
    use crate::image::raster::Memory;

    if !cache.contains(handle) {
        if cfg!(target_arch = "wasm32") {
            // TODO: Concurrent support for Wasm
            cache.insert(handle, Memory::load(handle));
        } else if let core::image::Handle::Rgba { .. } = handle {
            // Load RGBA handles synchronously, since it's very cheap
            cache.insert(handle, Memory::load(handle));
        } else if !pending.contains_key(&handle.id()) {
            let _ = pending.insert(handle.id(), Vec::from_iter(callback));

            #[cfg(not(target_arch = "wasm32"))]
            worker.load(handle);
        }
    }

    cache.get_mut(handle)
}

#[cfg(all(feature = "image", not(target_arch = "wasm32")))]
mod worker {
    use crate::core::image;
    use crate::graphics::Shell;
    use crate::image::atlas::{self, Atlas};
    use crate::image::raster;

    use std::sync::Arc;
    use std::sync::mpsc;
    use std::thread;

    pub struct Worker {
        jobs: mpsc::SyncSender<Job>,
        quit: mpsc::SyncSender<()>,
        work: mpsc::Receiver<Work>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl Worker {
        pub fn new(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            backend: wgpu::Backend,
            texture_layout: Arc<wgpu::BindGroupLayout>,
            shell: &Shell,
        ) -> Self {
            let (jobs_sender, jobs_receiver) = mpsc::sync_channel(1_000);
            let (quit_sender, quit_receiver) = mpsc::sync_channel(1);
            let (work_sender, work_receiver) = mpsc::sync_channel(1_000);

            let instance = Instance {
                device: device.clone(),
                queue: queue.clone(),
                backend,
                texture_layout,
                shell: shell.clone(),
                belt: wgpu::util::StagingBelt::new(4 * 1024 * 1024),
                jobs: jobs_receiver,
                output: work_sender,
                quit: quit_receiver,
            };

            let handle = thread::spawn(move || instance.run());

            Self {
                jobs: jobs_sender,
                quit: quit_sender,
                work: work_receiver,
                handle: Some(handle),
            }
        }

        pub fn load(&self, handle: &image::Handle) {
            let _ = self.jobs.send(Job::Load(handle.clone()));
        }

        pub fn upload(&self, handle: &image::Handle, image: raster::Image) {
            let _ = self.jobs.send(Job::Upload {
                handle: handle.clone(),
                width: image.width(),
                height: image.height(),
                rgba: image.into_raw(),
            });
        }

        pub fn drop(&self, bind_group: Arc<wgpu::BindGroup>) {
            let _ = self.jobs.send(Job::Drop(bind_group));
        }

        pub fn try_recv(&self) -> Result<Work, mpsc::TryRecvError> {
            self.work.try_recv()
        }

        pub fn quit(&mut self) {
            let _ = self.quit.try_send(());
            let _ = self.jobs.send(Job::Quit);
            let _ = self.handle.take().map(thread::JoinHandle::join);
        }
    }

    pub struct Instance {
        device: wgpu::Device,
        queue: wgpu::Queue,
        backend: wgpu::Backend,
        texture_layout: Arc<wgpu::BindGroupLayout>,
        shell: Shell,
        belt: wgpu::util::StagingBelt,
        jobs: mpsc::Receiver<Job>,
        output: mpsc::SyncSender<Work>,
        quit: mpsc::Receiver<()>,
    }

    #[derive(Debug)]
    enum Job {
        Load(image::Handle),
        Upload {
            handle: image::Handle,
            rgba: image::Bytes,
            width: u32,
            height: u32,
        },
        Drop(Arc<wgpu::BindGroup>),
        Quit,
    }

    pub enum Work {
        Upload {
            handle: image::Handle,
            entry: atlas::Entry,
            bind_group: Arc<wgpu::BindGroup>,
        },
        Error {
            handle: image::Handle,
            error: image::Error,
        },
    }

    impl Instance {
        fn run(mut self) {
            loop {
                if self.quit.try_recv().is_ok() {
                    return;
                }

                let Ok(job) = self.jobs.recv() else {
                    return;
                };

                match job {
                    Job::Load(handle) => {
                        match crate::graphics::image::load(&handle) {
                            Ok(image) => self.upload(
                                handle,
                                image.width(),
                                image.height(),
                                image.into_raw(),
                                Shell::invalidate_layout,
                            ),
                            Err(error) => {
                                let _ = self
                                    .output
                                    .send(Work::Error { handle, error });
                            }
                        }
                    }
                    Job::Upload {
                        handle,
                        rgba,
                        width,
                        height,
                    } => {
                        self.upload(
                            handle,
                            width,
                            height,
                            rgba,
                            Shell::request_redraw,
                        );
                    }
                    Job::Drop(bind_group) => {
                        drop(bind_group);
                    }
                    Job::Quit => return,
                }
            }
        }

        fn upload(
            &mut self,
            handle: image::Handle,
            width: u32,
            height: u32,
            rgba: image::Bytes,
            callback: fn(&Shell),
        ) {
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("raster image upload"),
                },
            );

            let mut atlas = Atlas::with_size(
                &self.device,
                self.backend,
                self.texture_layout.clone(),
                width.max(height),
            );

            let Some(entry) = atlas.upload(
                &self.device,
                &mut encoder,
                &mut self.belt,
                width,
                height,
                &rgba,
            ) else {
                return;
            };

            let output = self.output.clone();
            let shell = self.shell.clone();

            self.belt.finish();
            let submission = self.queue.submit([encoder.finish()]);
            self.belt.recall();

            let bind_group = atlas.bind_group().clone();

            self.queue.on_submitted_work_done(move || {
                let _ = output.send(Work::Upload {
                    handle,
                    entry,
                    bind_group,
                });

                callback(&shell);
            });

            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            });
        }
    }
}

impl Cache {
    fn region_from_entry(entry: &atlas::Entry) -> AtlasRegion {
        match entry {
            atlas::Entry::Contiguous(allocation) => {
                let (x, y) = allocation.position();
                let size = allocation.size();
                let layer = allocation.layer() as u32;
                let atlas_size = allocation.atlas_size() as f32;

                AtlasRegion {
                    uv_min: [x as f32 / atlas_size, y as f32 / atlas_size],
                    uv_max: [
                        (x + size.width) as f32 / atlas_size,
                        (y + size.height) as f32 / atlas_size,
                    ],
                    layer,
                }
            }
            atlas::Entry::Fragmented { size, fragments } => {
                if let Some(first) = fragments.first() {
                    let (x, y) = first.position;
                    let layer = first.allocation.layer() as u32;
                    let atlas_size = first.allocation.atlas_size() as f32;

                    AtlasRegion {
                        uv_min: [x as f32 / atlas_size, y as f32 / atlas_size],
                        uv_max: [
                            (x + size.width) as f32 / atlas_size,
                            (y + size.height) as f32 / atlas_size,
                        ],
                        layer,
                    }
                } else {
                    AtlasRegion {
                        uv_min: [0.0, 0.0],
                        uv_max: [0.0, 0.0],
                        layer: 0,
                    }
                }
            }
        }
    }
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use super::*;
    use crate::core::image::Handle;
    use crate::graphics::Shell;

    fn create_device() -> (wgpu::Device, wgpu::Queue, wgpu::Backend) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .expect("request adapter");

        let backend = adapter.get_info().backend;

        let descriptor = wgpu::DeviceDescriptor {
            label: Some("batched-image-test-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            ..Default::default()
        };

        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor))
                .expect("request device");

        (device, queue, backend)
    }

    fn create_layout(device: &wgpu::Device) -> Arc<wgpu::BindGroupLayout> {
        Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("test atlas layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: true,
                        },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                }],
            },
        ))
    }

    fn solid_handle(seed: u8) -> Handle {
        const WIDTH: u32 = 4;
        const HEIGHT: u32 = 4;

        let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);

        for _ in 0..(WIDTH * HEIGHT) {
            pixels.extend_from_slice(&[seed, 255 - seed, seed / 2, 255]);
        }

        Handle::from_rgba(WIDTH, HEIGHT, pixels)
    }

    #[test]
    fn cached_regions_survive_trim_after_hit() {
        let (device, queue, backend) = create_device();
        let layout = create_layout(&device);
        let shell = Shell::headless();
        let mut cache = Cache::new(&device, &queue, backend, layout, &shell);

        let handle = solid_handle(64);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batched-image-test-encoder"),
            });

        let mut belt = wgpu::util::StagingBelt::new(2 * 1024 * 1024);
        assert!(
            cache
                .ensure_raster_region(&device, &mut encoder, &mut belt, &handle)
                .is_some()
        );

        belt.finish();
        let submission = queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });
        belt.recall();

        // First frame trim: retains the allocation and resets cache bookkeeping.
        cache.trim();

        // Simulate a second frame: upload a new handle while touching the old
        // one to register a cache hit before the next trim.
        let new_handle = solid_handle(192);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batched-image-test-encoder-second"),
            });

        assert!(
            cache
                .ensure_raster_region(
                    &device,
                    &mut encoder,
                    &mut belt,
                    &new_handle
                )
                .is_some()
        );

        belt.finish();
        let submission = queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });
        belt.recall();

        assert!(cache.cached_raster_region(&handle).is_some());
        assert!(cache.cached_raster_region(&new_handle).is_some());

        cache.trim();

        assert!(cache.cached_raster_region(&handle).is_some());
        assert!(cache.cached_raster_region(&new_handle).is_some());
    }
}
