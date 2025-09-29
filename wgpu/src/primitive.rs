//! Draw custom primitives.
use crate::core::{self, Rectangle};
use crate::graphics::Viewport;
use crate::graphics::futures::{MaybeSend, MaybeSync};

use rustc_hash::{FxHashMap, FxHashSet};
use std::any::{Any, TypeId};
use std::fmt::Debug;

pub mod batch_state;
pub mod buffer_manager;
pub mod type_registry;

pub use batch_state::{
    AnyBatchState, BatchResources, BatchResourcesMut, PrepareContext,
    PrimitiveBatchState, RenderContext,
};
pub use buffer_manager::{BufferMemoryStats, InstanceBufferManager};
pub use type_registry::{
    BatchDescriptor, BatchEncodeContext, BatchRegistryGuard, descriptor,
    is_type_batchable, is_type_id_batchable, register_batchable_type,
    unregister_batchable_type,
};

/// A batch of primitives.
pub type Batch = Vec<Instance>;

/// A set of methods which allows a [`Primitive`] to be rendered.
pub trait Primitive: Debug + MaybeSend + MaybeSync + 'static {
    /// The shared renderer of this [`Primitive`].
    ///
    /// Normally, this will contain a bunch of [`wgpu`] state; like
    /// a rendering pipeline, buffers, and textures.
    ///
    /// All instances of this [`Primitive`] type will share the same
    /// [`Renderer`].
    type Renderer: MaybeSend + MaybeSync;

    /// Initializes the [`Renderer`](Self::Renderer) of the [`Primitive`].
    ///
    /// This will only be called once, when the first [`Primitive`] of this kind
    /// is encountered.
    fn initialize(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Self::Renderer;

    /// Processes the [`Primitive`], allowing for GPU buffer allocation.
    fn prepare(
        &self,
        renderer: &mut Self::Renderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    );

    /// Draws the [`Primitive`] in the given [`wgpu::RenderPass`].
    ///
    /// When possible, this should be implemented over [`render`](Self::render)
    /// since reusing the existing render pass should be considerably more
    /// efficient than issuing a new one.
    ///
    /// The viewport and scissor rect of the render pass provided is set
    /// to the bounds and clip bounds of the [`Primitive`], respectively.
    ///
    /// If you have complex composition needs, then you can leverage
    /// [`render`](Self::render) by returning `false` here.
    ///
    /// By default, it does nothing and returns `false`.
    fn draw(
        &self,
        _renderer: &Self::Renderer,
        _render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        false
    }

    /// Renders the [`Primitive`], using the given [`wgpu::CommandEncoder`].
    ///
    /// This will only be called if [`draw`](Self::draw) returns `false`.
    ///
    /// By default, it does nothing.
    fn render(
        &self,
        _renderer: &Self::Renderer,
        _encoder: &mut wgpu::CommandEncoder,
        _target: &wgpu::TextureView,
        _clip_bounds: &Rectangle<u32>,
    ) {
    }
}

/// Trait implemented by primitives that can be aggregated into GPU batches.
pub trait BatchPrimitive: Primitive {
    /// Renderer-managed state that stores the batching resources.
    type BatchState: PrimitiveBatchState;

    /// Creates the batch state the first time this primitive type is encountered.
    fn create_batch_state(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Self::BatchState {
        Self::BatchState::new(device, format)
    }

    /// Attempts to encode the primitive into the shared batch state.
    ///
    /// Returns `true` when the primitive has been successfully batched and the
    /// regular `prepare` path can be skipped for this frame.
    fn encode_batch(
        &self,
        state: &mut Self::BatchState,
        context: &BatchEncodeContext<'_>,
    ) -> bool;
}

pub(crate) trait Stored:
    Debug + MaybeSend + MaybeSync + 'static
{
    fn prepare(
        &self,
        storage: &mut Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        bounds: &Rectangle,
        viewport: &Viewport,
    );

    fn draw(
        &self,
        storage: &Storage,
        render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool;

    fn render(
        &self,
        storage: &Storage,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    );
}

#[derive(Debug)]
struct BlackBox<P: Primitive> {
    primitive: P,
}

impl<P: Primitive> Stored for BlackBox<P> {
    fn prepare(
        &self,
        storage: &mut Storage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        if storage.try_batch::<P>(
            &self.primitive,
            BatchEncodeContext::new(device, queue, format, bounds, viewport),
        ) {
            return;
        }

        if !storage.has::<P>() {
            storage.store::<P, _>(
                self.primitive.initialize(device, queue, format),
            );
        }

        let renderer = storage
            .get_mut::<P>()
            .expect("renderer should be initialized")
            .downcast_mut::<P::Renderer>()
            .expect("renderer should have the proper type");

        self.primitive
            .prepare(renderer, device, queue, bounds, viewport);
    }

    fn draw(
        &self,
        storage: &Storage,
        render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if storage.is_type_batched(&TypeId::of::<P>()) {
            return true;
        }

        let renderer = storage
            .get::<P>()
            .expect("renderer should be initialized")
            .downcast_ref::<P::Renderer>()
            .expect("renderer should have the proper type");

        self.primitive.draw(renderer, render_pass)
    }

    fn render(
        &self,
        storage: &Storage,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        if storage.is_type_batched(&TypeId::of::<P>()) {
            return;
        }

        let renderer = storage
            .get::<P>()
            .expect("renderer should be initialized")
            .downcast_ref::<P::Renderer>()
            .expect("renderer should have the proper type");

        self.primitive
            .render(renderer, encoder, target, clip_bounds);
    }
}

#[derive(Debug)]
/// An instance of a specific [`Primitive`].
pub struct Instance {
    /// The bounds of the [`Instance`].
    pub(crate) bounds: Rectangle,

    /// The [`Primitive`] to render.
    pub(crate) primitive: Box<dyn Stored>,
}

impl Instance {
    /// Creates a new [`Instance`] with the given [`Primitive`].
    pub fn new(bounds: Rectangle, primitive: impl Primitive) -> Self {
        Instance {
            bounds,
            primitive: Box::new(BlackBox { primitive }),
        }
    }
}

/// A renderer than can draw custom primitives.
pub trait Renderer: core::Renderer {
    /// Draws a custom primitive.
    fn draw_primitive(&mut self, bounds: Rectangle, primitive: impl Primitive);
}

/// Stores custom, user-provided types and their renderer-managed state.
#[derive(Default)]
pub struct Storage {
    pipelines: FxHashMap<TypeId, Box<dyn AnyConcurrent>>,
    batch_states: FxHashMap<TypeId, Box<dyn AnyBatchState>>,
    active_batches: FxHashSet<TypeId>,
    layer_segments: Vec<LayerSegment>,
    layer_markers: FxHashMap<TypeId, usize>,
}

#[derive(Debug)]
struct LayerSegment {
    layer: usize,
    type_id: TypeId,
    range: std::ops::Range<u32>,
}

impl Storage {
    /// Returns `true` if `Storage` contains a type `T`.
    pub fn has<T: 'static>(&self) -> bool {
        self.pipelines.contains_key(&TypeId::of::<T>())
    }

    /// Inserts the data `T` in to [`Storage`].
    pub fn store<T: 'static, D: Any + MaybeSend + MaybeSync>(
        &mut self,
        data: D,
    ) {
        let _ = self.pipelines.insert(TypeId::of::<T>(), Box::new(data));
    }

    /// Returns a reference to the data with type `T` if it exists in [`Storage`].
    pub fn get<T: 'static>(&self) -> Option<&dyn Any> {
        self.pipelines
            .get(&TypeId::of::<T>())
            .map(|pipeline| pipeline.as_ref() as &dyn Any)
    }

    /// Returns a mutable reference to the data with type `T` if it exists in [`Storage`].
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut dyn Any> {
        self.pipelines
            .get_mut(&TypeId::of::<T>())
            .map(|pipeline| pipeline.as_mut() as &mut dyn Any)
    }

    /// Attempts to batch the given primitive using the registered descriptor, if any.
    pub fn try_batch<P: Primitive>(
        &mut self,
        primitive: &P,
        context: BatchEncodeContext<'_>,
    ) -> bool {
        let type_id = TypeId::of::<P>();

        let Some(descriptor_guard) = type_registry::descriptor(&type_id) else {
            return false;
        };

        let descriptor = descriptor_guard.descriptor();
        let primitive_any = primitive as &dyn Any;

        let batched = {
            let state = self.ensure_batch_state(
                type_id,
                descriptor,
                context.device,
                context.format,
            );
            (descriptor.encode_instance)(primitive_any, state, &context)
        };

        if batched {
            self.mark_batched(type_id);
        }

        batched
    }

    /// Returns true when a batch state already exists for `type_id`.
    pub fn has_batch_state(&self, type_id: &TypeId) -> bool {
        self.batch_states.contains_key(type_id)
    }

    /// Marks a primitive type as having produced batched work during the current frame.
    pub fn mark_batched(&mut self, type_id: TypeId) {
        let _ = self.active_batches.insert(type_id);
    }

    /// Clears per-frame layer tracking state. Must be called once at the start of a frame.
    pub fn begin_frame(&mut self) {
        self.layer_segments.clear();
        self.layer_markers.clear();
    }

    /// Records the instance counts at the start of a layer so per-layer ranges can be computed.
    pub fn begin_layer(&mut self, _layer_index: usize) {
        self.layer_markers.clear();

        for (type_id, state) in &self.batch_states {
            if let Some(descriptor_guard) = type_registry::descriptor(type_id) {
                let descriptor = descriptor_guard.descriptor();
                let count = (descriptor.instance_count)(state.as_ref());
                let _ = self.layer_markers.insert(*type_id, count);
            }
        }
    }

    /// Stores the per-layer instance range for each active batch type.
    pub fn end_layer(&mut self, layer_index: usize) {
        for (type_id, state) in &self.batch_states {
            let Some(descriptor_guard) = type_registry::descriptor(type_id)
            else {
                continue;
            };

            let descriptor = descriptor_guard.descriptor();
            let end = (descriptor.instance_count)(state.as_ref());
            let start = self.layer_markers.get(type_id).copied().unwrap_or(0);

            if end > start {
                self.layer_segments.push(LayerSegment {
                    layer: layer_index,
                    type_id: *type_id,
                    range: (start as u32)..(end as u32),
                });
            }
        }
    }

    /// Returns whether a primitive type has been batched in the current frame.
    pub fn is_type_batched(&self, type_id: &TypeId) -> bool {
        self.active_batches.contains(type_id)
    }

    fn ensure_batch_state(
        &mut self,
        type_id: TypeId,
        descriptor: &BatchDescriptor,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &mut dyn AnyBatchState {
        self.batch_states
            .entry(type_id)
            .or_insert_with(|| (descriptor.create_state)(device, format))
            .as_mut()
    }

    /// Prepares all active batch states for rendering.
    pub fn prepare_batches(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        resources: &mut BatchResourcesMut<'_>,
        viewport: &Viewport,
        scale_factor: f32,
    ) {
        let active_ids: Vec<TypeId> =
            self.active_batches.iter().cloned().collect();

        for type_id in active_ids {
            let Some(state) = self.batch_states.get_mut(&type_id) else {
                continue;
            };

            let Some(descriptor_guard) = type_registry::descriptor(&type_id)
            else {
                continue;
            };

            let descriptor = descriptor_guard.descriptor();
            let instance_count = (descriptor.instance_count)(state.as_ref());

            if instance_count == 0 {
                log::warn!(
                    "Active batch {:?} reported zero instances before prepare; skipping",
                    type_id
                );
                continue;
            }

            if (descriptor.is_empty)(state.as_ref()) {
                continue;
            }

            let mut context = PrepareContext {
                device,
                encoder,
                belt,
                resources: resources.reborrow(),
                viewport,
                scale_factor,
            };

            (descriptor.prepare_batches)(state.as_mut(), &mut context);
        }
    }

    /// Renders all active batch states.
    pub fn render_batches(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'_>,
        resources: &BatchResources<'_>,
        scissor_rect: Rectangle<u32>,
    ) {
        for type_id in self.active_batches.iter() {
            let Some(state) = self.batch_states.get(type_id) else {
                continue;
            };

            let Some(descriptor_guard) = type_registry::descriptor(type_id)
            else {
                continue;
            };

            let descriptor = descriptor_guard.descriptor();
            let instance_count = (descriptor.instance_count)(state.as_ref());

            if instance_count == 0 {
                log::warn!(
                    "Active batch {:?} reported zero instances before render; skipping",
                    type_id
                );
                continue;
            }

            if (descriptor.is_empty)(state.as_ref()) {
                continue;
            }

            let mut context = RenderContext {
                resources: resources.reborrow(),
                scissor_rect,
            };

            (descriptor.render_batches)(
                state.as_ref(),
                render_pass,
                &mut context,
                0..(instance_count as u32),
            );
        }
    }

    /// Renders all batched work recorded for the given layer and consumes the associated segments.
    pub fn render_layer_batches(
        &mut self,
        layer_index: usize,
        render_pass: &mut wgpu::RenderPass<'_>,
        resources: &BatchResources<'_>,
        scissor_rect: Rectangle<u32>,
    ) {
        let mut index = 0;

        while index < self.layer_segments.len() {
            if self.layer_segments[index].layer != layer_index {
                index += 1;
                continue;
            }

            let segment = self.layer_segments.remove(index);

            let Some(state) = self.batch_states.get(&segment.type_id) else {
                continue;
            };

            let Some(descriptor_guard) =
                type_registry::descriptor(&segment.type_id)
            else {
                continue;
            };

            let descriptor = descriptor_guard.descriptor();

            if (descriptor.is_empty)(state.as_ref()) {
                continue;
            }

            if segment.range.start >= segment.range.end {
                continue;
            }

            let mut context = RenderContext {
                resources: resources.reborrow(),
                scissor_rect,
            };

            (descriptor.render_batches)(
                state.as_ref(),
                render_pass,
                &mut context,
                segment.range.clone(),
            );
        }
    }

    /// Discards pending batch segments recorded for the given layer without rendering them.
    pub fn drop_layer_segments(&mut self, layer_index: usize) {
        self.layer_segments
            .retain(|segment| segment.layer != layer_index);
    }

    /// Returns true when there are still layer segments awaiting rendering.
    pub fn has_pending_layer_segments(&self) -> bool {
        !self.layer_segments.is_empty()
    }

    /// Clears per-frame batch metadata while preserving allocations.
    pub fn trim_batches(&mut self) {
        for type_id in self.active_batches.drain() {
            if let Some(state) = self.batch_states.get_mut(&type_id)
                && let Some(descriptor_guard) =
                    type_registry::descriptor(&type_id)
            {
                let descriptor = descriptor_guard.descriptor();
                (descriptor.trim_batches)(state.as_mut());
            }
        }

        self.layer_segments.clear();
        self.layer_markers.clear();
    }

    /// Returns true when any active batch state contains work to render.
    pub fn has_non_empty_batches(&self) -> bool {
        self.active_batches.iter().any(|type_id| {
            self.batch_states
                .get(type_id)
                .and_then(|state| {
                    type_registry::descriptor(type_id).map(|guard| {
                        let descriptor = guard.descriptor();
                        !(descriptor.is_empty)(state.as_ref())
                    })
                })
                .unwrap_or(false)
        })
    }
}

trait AnyConcurrent: Any + MaybeSend + MaybeSync {}

impl<T> AnyConcurrent for T where T: Any + MaybeSend + MaybeSync {}
