use batched_image::batched_image_widget;
use iced::widget::image::Handle;
use iced::widget::{Column, Row, text};
use iced::{Alignment, Element, Length};

const TILE_SIZE: f32 = 128.0;
const GRID_COLUMNS: usize = 4;
const TILE_COUNT: usize = 12;

fn main() -> iced::Result {
    iced::application(Example::default, Example::update, Example::view).run()
}

struct Example {
    tiles: Vec<Tile>,
}

#[derive(Clone)]
struct Tile {
    id: u64,
    handle: Handle,
}

impl Example {
    fn update(&mut self, _message: ()) {}

    fn view(&self) -> Element<'_, ()> {
        let header = text(
            "First frame uploads textures; subsequent frames reuse the atlas.",
        )
        .size(20);

        let mut grid = Column::new().spacing(16);

        for chunk in self.tiles.chunks(GRID_COLUMNS) {
            let row = chunk.iter().fold(Row::new().spacing(16), |row, tile| {
                row.push(batched_image_widget(
                    tile.handle.clone(),
                    tile.id,
                    Length::Fixed(TILE_SIZE),
                ))
            });

            grid = grid.push(row.align_y(Alignment::Center));
        }

        Column::new()
            .push(header)
            .push(grid)
            .padding(32)
            .spacing(32)
            .align_x(Alignment::Center)
            .into()
    }
}

impl Default for Example {
    fn default() -> Self {
        Self {
            tiles: generate_tiles(),
        }
    }
}

fn generate_tiles() -> Vec<Tile> {
    (0..TILE_COUNT)
        .map(|index| Tile {
            id: index as u64,
            handle: gradient_handle(index as u8),
        })
        .collect()
}

fn gradient_handle(seed: u8) -> Handle {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let xf = x as f32 / WIDTH as f32;
            let yf = y as f32 / HEIGHT as f32;

            let r = ((seed as f32 / TILE_COUNT as f32) * 255.0) as u8;
            let g = (xf * 255.0) as u8;
            let b = (yf * 255.0) as u8;

            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }

    Handle::from_rgba(WIDTH, HEIGHT, pixels)
}

mod batched_image {
    use bytemuck::{Pod, Zeroable};
    use iced::advanced::graphics::Viewport;
    use iced::wgpu;
    use iced::widget::image::Handle;
    use iced::widget::shader;
    use iced::widget::shader::{self as shader_widget, Program};
    use iced::{Length, Rectangle, mouse};
    use iced_wgpu::AtlasRegion;
    use iced_wgpu::primitive::{
        BatchEncodeContext, BatchPrimitive, PrepareContext,
        PrimitiveBatchState, RenderContext,
        buffer_manager::InstanceBufferManager, register_batchable_type,
    };
    use std::collections::VecDeque;
    use std::sync::{Arc, OnceLock};

    const MAX_UPLOADS_PER_FRAME: u32 = 8;

    pub fn batched_image_widget(
        handle: Handle,
        id: u64,
        size: Length,
    ) -> iced::Element<'static, ()> {
        shader(ImageProgram { id, handle })
            .width(size)
            .height(size)
            .into()
    }

    #[derive(Clone)]
    struct ImageProgram {
        id: u64,
        handle: Handle,
    }

    impl Program<()> for ImageProgram {
        type State = ();
        type Primitive = ImagePrimitive;

        fn draw(
            &self,
            _state: &Self::State,
            _cursor: mouse::Cursor,
            bounds: Rectangle,
        ) -> Self::Primitive {
            ensure_batch_registration();

            ImagePrimitive {
                id: self.id,
                handle: self.handle.clone(),
                bounds,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct ImagePrimitive {
        id: u64,
        handle: Handle,
        bounds: Rectangle,
    }

    impl shader_widget::Primitive for ImagePrimitive {
        type Renderer = ();

        fn initialize(
            &self,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _format: wgpu::TextureFormat,
        ) -> Self::Renderer {
        }

        fn prepare(
            &self,
            _renderer: &mut Self::Renderer,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _bounds: &Rectangle,
            _viewport: &Viewport,
        ) {
        }
    }

    impl BatchPrimitive for ImagePrimitive {
        type BatchState = ImageBatchState;

        fn create_batch_state(
            device: &wgpu::Device,
            format: wgpu::TextureFormat,
        ) -> Self::BatchState {
            ImageBatchState::new(device, format)
        }

        fn encode_batch(
            &self,
            state: &mut Self::BatchState,
            _context: &BatchEncodeContext<'_>,
        ) -> bool {
            state.enqueue(PendingPrimitive {
                id: self.id,
                handle: self.handle.clone(),
                bounds: self.bounds,
            });

            true
        }
    }

    #[derive(Clone)]
    struct PendingPrimitive {
        id: u64,
        handle: Handle,
        bounds: Rectangle,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct ImageInstance {
        position_and_size: [f32; 4],
        atlas_uvs: [f32; 4],
        layer_opacity: [f32; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Globals {
        transform: [f32; 16],
        scale_factor: f32,
        _padding: [f32; 3],
    }

    struct ImageBatchState {
        pending: VecDeque<PendingPrimitive>,
        staged_instances: Vec<ImageInstance>,
        instance_manager: InstanceBufferManager<ImageInstance>,
        render_pipeline: Option<Arc<wgpu::RenderPipeline>>,
        shader: Arc<wgpu::ShaderModule>,
        atlas_layout: Option<Arc<wgpu::BindGroupLayout>>,
        surface_format: wgpu::TextureFormat,
        globals_buffer: Option<wgpu::Buffer>,
        globals_bind_group: Option<wgpu::BindGroup>,
        globals_layout: Arc<wgpu::BindGroupLayout>,
        sampler: Arc<wgpu::Sampler>,
        uploads_this_frame: u32,
    }

    impl ImageBatchState {
        fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
            let shader = Arc::new(device.create_shader_module(
                wgpu::ShaderModuleDescriptor {
                    label: Some("batched image shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("image.wgsl").into(),
                    ),
                },
            ));

            let globals_layout = Arc::new(
                device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        label: Some("batched image globals"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::VERTEX,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: Some(
                                        wgpu::BufferSize::new(
                                            std::mem::size_of::<Globals>()
                                                as u64,
                                        )
                                        .unwrap(),
                                    ),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(
                                    wgpu::SamplerBindingType::Filtering,
                                ),
                                count: None,
                            },
                        ],
                    },
                ),
            );

            let sampler =
                Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("batched image sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                }));

            Self {
                pending: VecDeque::new(),
                staged_instances: Vec::new(),
                instance_manager: InstanceBufferManager::new(),
                render_pipeline: None,
                shader,
                atlas_layout: None,
                surface_format: format,
                globals_buffer: None,
                globals_bind_group: None,
                globals_layout,
                sampler,
                uploads_this_frame: 0,
            }
        }

        fn enqueue(&mut self, primitive: PendingPrimitive) {
            if let Some(position) = self
                .pending
                .iter()
                .position(|existing| existing.id == primitive.id)
            {
                let _ = self.pending.remove(position);
            }

            self.pending.push_back(primitive);
        }

        fn ensure_pipeline(
            &mut self,
            device: &wgpu::Device,
            atlas_layout: Arc<wgpu::BindGroupLayout>,
        ) {
            if self
                .atlas_layout
                .as_ref()
                .is_some_and(|existing| Arc::ptr_eq(existing, &atlas_layout))
                && self.render_pipeline.is_some()
            {
                return;
            }

            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("batched image pipeline layout"),
                    bind_group_layouts: &[
                        &self.globals_layout,
                        atlas_layout.as_ref(),
                    ],
                    push_constant_ranges: &[],
                },
            );

            let vertex_layout = wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ImageInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 32,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            };

            let pipeline = device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("batched image pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &self.shader,
                        entry_point: Some("vs_main"),
                        buffers: &[vertex_layout],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &self.shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.surface_format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                },
            );

            self.render_pipeline = Some(Arc::new(pipeline));
            self.atlas_layout = Some(atlas_layout);
        }

        fn stage_instance(&mut self, region: AtlasRegion, bounds: &Rectangle) {
            let instance = ImageInstance {
                position_and_size: [
                    bounds.x,
                    bounds.y,
                    bounds.width,
                    bounds.height,
                ],
                atlas_uvs: [
                    region.uv_min[0],
                    region.uv_min[1],
                    region.uv_max[0],
                    region.uv_max[1],
                ],
                layer_opacity: [region.layer as f32, 1.0, 0.0, 0.0],
            };

            self.staged_instances.push(instance);
        }
    }

    impl std::fmt::Debug for ImageBatchState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ImageBatchState")
                .field(
                    "uploaded_instances",
                    &self.instance_manager.instance_count(),
                )
                .finish()
        }
    }

    impl PrimitiveBatchState for ImageBatchState {
        type InstanceData = ImageInstance;

        fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
            Self::new(device, format)
        }

        fn add_instance(&mut self, instance: Self::InstanceData) {
            self.staged_instances.push(instance);
        }

        fn prepare(&mut self, context: &mut PrepareContext<'_>) {
            if let Some(image_cache) = context.resources.image_cache() {
                let atlas_layout = image_cache.texture_layout();
                self.ensure_pipeline(context.device, atlas_layout);

                let pending_len = self.pending.len();

                for _ in 0..pending_len {
                    let Some(pending) = self.pending.pop_front() else {
                        break;
                    };

                    let mut atlas_region =
                        image_cache.cached_raster_region(&pending.handle);

                    if atlas_region.is_none() {
                        if self.uploads_this_frame >= MAX_UPLOADS_PER_FRAME {
                            self.pending.push_back(pending);
                            continue;
                        }

                        if let Some(region) = image_cache.ensure_raster_region(
                            context.device,
                            context.encoder,
                            context.belt,
                            &pending.handle,
                        ) {
                            self.uploads_this_frame += 1;
                            atlas_region = Some(region);
                        } else {
                            self.pending.push_back(pending);
                            continue;
                        }
                    }

                    if let Some(region) = atlas_region {
                        self.stage_instance(region, &pending.bounds);
                    }
                }
            }

            for instance in self.staged_instances.drain(..) {
                self.instance_manager.add_instance(instance);
            }

            if self
                .instance_manager
                .upload(context.device, context.encoder, context.belt)
                .is_none()
            {
                return;
            }

            let globals = Globals {
                transform: context.viewport.projection().into(),
                scale_factor: context.scale_factor,
                _padding: [0.0; 3],
            };

            if self.globals_buffer.is_none() {
                self.globals_buffer = Some(context.device.create_buffer(
                    &wgpu::BufferDescriptor {
                        label: Some("batched image globals"),
                        size: std::mem::size_of::<Globals>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                ));
            }

            if let Some(buffer) = &self.globals_buffer {
                context
                    .belt
                    .write_buffer(
                        context.encoder,
                        buffer,
                        0,
                        wgpu::BufferSize::new(
                            std::mem::size_of::<Globals>() as u64
                        )
                        .unwrap(),
                        context.device,
                    )
                    .copy_from_slice(bytemuck::bytes_of(&globals));

                if self.globals_bind_group.is_none() {
                    self.globals_bind_group =
                        Some(context.device.create_bind_group(
                            &wgpu::BindGroupDescriptor {
                                label: Some("batched image globals bind group"),
                                layout: &self.globals_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource:
                                            wgpu::BindingResource::Sampler(
                                                &self.sampler,
                                            ),
                                    },
                                ],
                            },
                        ));
                }
            }
        }

        fn render(
            &self,
            render_pass: &mut wgpu::RenderPass<'_>,
            context: &mut RenderContext<'_>,
            range: std::ops::Range<u32>,
        ) {
            let count = self.instance_manager.instance_count() as u32;
            if count == 0 {
                return;
            }

            let start = range.start.min(count);
            let end = range.end.min(count);
            if start >= end {
                return;
            }

            let Some(image_cache) = context.resources.image_cache() else {
                return;
            };

            let (Some(instance_buffer), Some(globals), Some(pipeline)) = (
                self.instance_manager.buffer(),
                self.globals_bind_group.as_ref(),
                self.render_pipeline.as_ref(),
            ) else {
                return;
            };

            let atlas_bind_group = image_cache.bind_group();

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, globals, &[]);
            render_pass.set_bind_group(1, atlas_bind_group.as_ref(), &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, start..end);
        }

        fn trim(&mut self) {
            self.instance_manager.clear();
            self.pending.clear();
            self.staged_instances.clear();
            self.uploads_this_frame = 0;
        }

        fn instance_count(&self) -> usize {
            self.instance_manager.instance_count()
        }
    }

    static REGISTRATION: OnceLock<()> = OnceLock::new();

    fn ensure_batch_registration() {
        REGISTRATION.get_or_init(|| {
            register_batchable_type::<ImagePrimitive>();
        });
    }
}
