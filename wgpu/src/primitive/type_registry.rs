//! Global registry describing how each batchable primitive type should be
//! encoded, prepared, and rendered.

use super::BatchPrimitive;
use super::batch_state::{
    AnyBatchState, PrepareContext, PrimitiveBatchState, RenderContext,
};

use rustc_hash::FxHashMap;
use std::any::{Any, TypeId};
use std::sync::{OnceLock, RwLock, RwLockReadGuard};

use crate::core::Rectangle;
use crate::graphics::Viewport;

/// Context forwarded to `BatchPrimitive::encode_batch` implementations.
#[derive(Debug, Copy, Clone)]
pub struct BatchEncodeContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub format: wgpu::TextureFormat,
    pub bounds: &'a Rectangle,
    pub viewport: &'a Viewport,
}

impl<'a> BatchEncodeContext<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        format: wgpu::TextureFormat,
        bounds: &'a Rectangle,
        viewport: &'a Viewport,
    ) -> Self {
        Self {
            device,
            queue,
            format,
            bounds,
            viewport,
        }
    }
}

/// Function pointers describing how to handle a batchable primitive type.
#[derive(Debug)]
pub struct BatchDescriptor {
    pub(crate) create_state:
        fn(&wgpu::Device, wgpu::TextureFormat) -> Box<dyn AnyBatchState>,
    pub(crate) encode_instance:
        fn(&dyn Any, &mut dyn AnyBatchState, &BatchEncodeContext<'_>) -> bool,
    pub(crate) prepare_batches:
        fn(&mut dyn AnyBatchState, &mut PrepareContext<'_>),
    pub(crate) render_batches: fn(
        &dyn AnyBatchState,
        &mut wgpu::RenderPass<'_>,
        &mut RenderContext<'_>,
        std::ops::Range<u32>,
    ),
    pub(crate) trim_batches: fn(&mut dyn AnyBatchState),
    #[allow(dead_code)]
    pub(crate) instance_count: fn(&dyn AnyBatchState) -> usize,
    pub(crate) is_empty: fn(&dyn AnyBatchState) -> bool,
}

impl BatchDescriptor {
    pub fn new<P, S>() -> Self
    where
        P: BatchPrimitive<BatchState = S> + 'static,
        S: PrimitiveBatchState + 'static,
    {
        Self {
            create_state: create_state::<P, S>,
            encode_instance: encode_instance::<P, S>,
            prepare_batches: prepare_batches::<S>,
            render_batches: render_batches::<S>,
            trim_batches: trim_batches::<S>,
            instance_count: instance_count::<S>,
            is_empty: is_empty::<S>,
        }
    }
}

fn create_state<P, S>(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> Box<dyn AnyBatchState>
where
    P: BatchPrimitive<BatchState = S>,
    S: PrimitiveBatchState + 'static,
{
    Box::new(P::create_batch_state(device, format))
}

fn encode_instance<P, S>(
    primitive: &dyn Any,
    state: &mut dyn AnyBatchState,
    context: &BatchEncodeContext<'_>,
) -> bool
where
    P: BatchPrimitive<BatchState = S>,
    S: PrimitiveBatchState + 'static,
{
    let primitive = primitive
        .downcast_ref::<P>()
        .expect("batch descriptor primitive type mismatch");
    let state = state
        .as_any_mut()
        .downcast_mut::<S>()
        .expect("batch descriptor state type mismatch");

    primitive.encode_batch(state, context)
}

fn prepare_batches<S>(
    state: &mut dyn AnyBatchState,
    context: &mut PrepareContext<'_>,
) where
    S: PrimitiveBatchState + 'static,
{
    let state = state
        .as_any_mut()
        .downcast_mut::<S>()
        .expect("batch descriptor state type mismatch");

    state.prepare(context);
}

fn render_batches<S>(
    state: &dyn AnyBatchState,
    render_pass: &mut wgpu::RenderPass<'_>,
    context: &mut RenderContext<'_>,
    range: std::ops::Range<u32>,
) where
    S: PrimitiveBatchState + 'static,
{
    let state = state
        .as_any()
        .downcast_ref::<S>()
        .expect("batch descriptor state type mismatch");

    state.render(render_pass, context, range);
}

fn trim_batches<S>(state: &mut dyn AnyBatchState)
where
    S: PrimitiveBatchState + 'static,
{
    let state = state
        .as_any_mut()
        .downcast_mut::<S>()
        .expect("batch descriptor state type mismatch");

    state.trim();
}

fn instance_count<S>(state: &dyn AnyBatchState) -> usize
where
    S: PrimitiveBatchState + 'static,
{
    let state = state
        .as_any()
        .downcast_ref::<S>()
        .expect("batch descriptor state type mismatch");

    state.instance_count()
}

fn is_empty<S>(state: &dyn AnyBatchState) -> bool
where
    S: PrimitiveBatchState + 'static,
{
    let state = state
        .as_any()
        .downcast_ref::<S>()
        .expect("batch descriptor state type mismatch");

    state.is_empty()
}

#[derive(Debug, Default)]
pub struct TypeRegistry {
    descriptors: RwLock<FxHashMap<TypeId, BatchDescriptor>>,
}

impl TypeRegistry {
    pub fn register_descriptor(
        &self,
        type_id: TypeId,
        descriptor: BatchDescriptor,
    ) {
        let mut registry =
            self.descriptors.write().expect("lock batch registry");
        let _ = registry.insert(type_id, descriptor);
    }

    pub fn unregister(&self, type_id: &TypeId) {
        let mut registry =
            self.descriptors.write().expect("lock batch registry");
        let _ = registry.remove(type_id);
    }

    pub fn descriptor(
        &self,
        type_id: &TypeId,
    ) -> Option<BatchRegistryGuard<'_>> {
        let registry = self.descriptors.read().expect("lock batch registry");

        if registry.contains_key(type_id) {
            Some(BatchRegistryGuard {
                guard: registry,
                type_id: *type_id,
            })
        } else {
            None
        }
    }
}

pub struct BatchRegistryGuard<'a> {
    guard: RwLockReadGuard<'a, FxHashMap<TypeId, BatchDescriptor>>,
    type_id: TypeId,
}

impl<'a> BatchRegistryGuard<'a> {
    pub fn descriptor(&self) -> &BatchDescriptor {
        self.guard
            .get(&self.type_id)
            .expect("descriptor should exist while guard is alive")
    }
}

static GLOBAL_TYPE_REGISTRY: OnceLock<TypeRegistry> = OnceLock::new();

fn registry() -> &'static TypeRegistry {
    GLOBAL_TYPE_REGISTRY.get_or_init(TypeRegistry::default)
}

pub fn register_batchable_type<P>()
where
    P: BatchPrimitive,
    P::BatchState: PrimitiveBatchState + 'static,
{
    let descriptor = BatchDescriptor::new::<P, P::BatchState>();
    registry().register_descriptor(TypeId::of::<P>(), descriptor);
}

pub fn unregister_batchable_type<P>()
where
    P: BatchPrimitive,
{
    registry().unregister(&TypeId::of::<P>());
}

pub fn is_type_batchable<P>() -> bool
where
    P: 'static,
{
    is_type_id_batchable(&TypeId::of::<P>())
}

pub fn is_type_id_batchable(type_id: &TypeId) -> bool {
    registry().descriptor(type_id).is_some()
}

pub fn descriptor(type_id: &TypeId) -> Option<BatchRegistryGuard<'_>> {
    registry().descriptor(type_id)
}
