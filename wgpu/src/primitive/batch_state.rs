//! Batch state abstractions for aggregating and rendering multiple instances
//! of custom primitives in a single draw call.

use crate::core::Rectangle;
use crate::graphics;

use std::any::Any;
use std::fmt::{self, Debug};

use wgpu::util::StagingBelt;

#[cfg(any(feature = "image", feature = "svg"))]
type ImageCacheMut<'a> = &'a mut crate::image::Cache;
#[cfg(not(any(feature = "image", feature = "svg")))]
type ImageCacheMut<'a> = &'a mut ();

#[cfg(any(feature = "image", feature = "svg"))]
type ImageCacheRef<'a> = &'a crate::image::Cache;
#[cfg(not(any(feature = "image", feature = "svg")))]
type ImageCacheRef<'a> = &'a ();

/// Optional caches and resources that batched primitives may need while
/// preparing GPU work.
#[derive(Default)]
pub struct BatchResourcesMut<'a> {
    pub image_cache: Option<ImageCacheMut<'a>>,
}

impl<'a> BatchResourcesMut<'a> {
    /// Creates a reborrowed view of the underlying resources.
    pub fn reborrow(&mut self) -> BatchResourcesMut<'_> {
        BatchResourcesMut {
            image_cache: self.image_cache.as_deref_mut(),
        }
    }

    /// Returns a mutable reference to the image cache when available.
    ///
    /// When batching image primitives, callers should prefer
    /// [`crate::image::Cache::cached_raster_region`] over manual atlas lookups
    /// so cache hits are registered for the current frame. Registered hits allow
    /// trim passes to keep existing atlas allocations alive between frames.
    pub fn image_cache(&mut self) -> Option<ImageCacheMut<'_>> {
        self.image_cache.as_deref_mut()
    }
}

/// Immutable caches exposed to batched primitives during rendering.
#[derive(Default)]
pub struct BatchResources<'a> {
    pub image_cache: Option<ImageCacheRef<'a>>,
}

impl<'a> BatchResources<'a> {
    /// Creates a reborrowed view of the resources without extending borrows.
    pub fn reborrow(&self) -> BatchResources<'_> {
        BatchResources {
            image_cache: self.image_cache,
        }
    }

    /// Returns an immutable reference to the image cache when present.
    pub fn image_cache(&self) -> Option<ImageCacheRef<'_>> {
        self.image_cache
    }
}

impl<'a> Debug for BatchResourcesMut<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchResourcesMut")
            .field(
                "image_cache",
                &self
                    .image_cache
                    .as_ref()
                    .map(|_| "Some(Cache)")
                    .unwrap_or("None"),
            )
            .finish()
    }
}

impl<'a> Debug for BatchResources<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchResources")
            .field(
                "image_cache",
                &self
                    .image_cache
                    .as_ref()
                    .map(|_| "Some(Cache)")
                    .unwrap_or("None"),
            )
            .finish()
    }
}

/// Context passed to `PrimitiveBatchState::prepare` containing per-frame
/// resources required to upload batched data to the GPU.
#[derive(Debug)]
pub struct PrepareContext<'a> {
    pub device: &'a wgpu::Device,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub belt: &'a mut StagingBelt,
    pub resources: BatchResourcesMut<'a>,
    pub viewport: &'a graphics::Viewport,
    pub scale_factor: f32,
}

/// Context passed to `PrimitiveBatchState::render` containing immutable caches
/// and clipping metadata.
#[derive(Debug)]
pub struct RenderContext<'a> {
    pub resources: BatchResources<'a>,
    pub scissor_rect: Rectangle<u32>,
}

/// Trait for managing batched rendering of a specific custom primitive type.
pub trait PrimitiveBatchState: Debug + Send + Sync + 'static {
    /// The GPU instance data for this primitive type.
    type InstanceData: bytemuck::Pod + bytemuck::Zeroable;

    /// Creates a new instance of this batch state.
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self
    where
        Self: Sized;

    /// Accumulates an instance for batching.
    fn add_instance(&mut self, instance: Self::InstanceData);

    /// Prepares all accumulated instances for rendering.
    ///
    /// Implementations that interact with the image cache should call
    /// [`crate::image::Cache::cached_raster_region`] for textures that are
    /// already uploaded and fall back to `ensure_raster_region` for new ones.
    /// The cached lookup keeps atlas entries marked as “in use”, preventing the
    /// renderer's trim pass from evicting them on the next frame.
    fn prepare(&mut self, context: &mut PrepareContext<'_>);

    /// Renders a range of batched instances in a single draw call.
    fn render(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &mut RenderContext<'_>,
        range: std::ops::Range<u32>,
    );

    /// Clears per-frame accumulated data while keeping allocations.
    fn trim(&mut self);

    /// Returns the number of accumulated instances.
    fn instance_count(&self) -> usize;

    /// Returns whether the batch currently holds no instances.
    fn is_empty(&self) -> bool {
        self.instance_count() == 0
    }
}

/// Type-erased batch state for storage in heterogeneous collections.
pub trait AnyBatchState: Debug + Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn prepare_erased(&mut self, context: &mut PrepareContext<'_>);
    fn render_erased(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &mut RenderContext<'_>,
        range: std::ops::Range<u32>,
    );
    fn trim_erased(&mut self);
    fn is_empty_erased(&self) -> bool;
    fn instance_count_erased(&self) -> usize;
}

impl<T> AnyBatchState for T
where
    T: PrimitiveBatchState,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn prepare_erased(&mut self, context: &mut PrepareContext<'_>) {
        self.prepare(context);
    }

    fn render_erased(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        context: &mut RenderContext<'_>,
        range: std::ops::Range<u32>,
    ) {
        self.render(render_pass, context, range);
    }

    fn trim_erased(&mut self) {
        self.trim();
    }

    fn is_empty_erased(&self) -> bool {
        self.is_empty()
    }

    fn instance_count_erased(&self) -> usize {
        self.instance_count()
    }
}
