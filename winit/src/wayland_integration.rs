/// Wayland integration hooks for subsurface-aware compositors.
#[cfg(all(target_os = "linux", feature = "wayland-hack"))]
pub mod wayland {
    use std::cell::RefCell;
    use std::sync::{Arc, Mutex};
    use winit::raw_window_handle::{
        HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    };

    thread_local! {
        /// Thread-local storage for the active Wayland integration context.
        static CURRENT_INTEGRATION: RefCell<Option<WaylandIntegration>> =
            const { RefCell::new(None) };
    }

    /// Sets the active Wayland integration context for the current thread.
    pub fn set_current_wayland_integration(
        integration: Option<WaylandIntegration>,
    ) {
        CURRENT_INTEGRATION.with(|current| {
            *current.borrow_mut() = integration;
        });
    }

    /// Runs `f` with the current Wayland integration context, if any.
    pub fn with_current_wayland_integration<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&WaylandIntegration) -> R,
    {
        CURRENT_INTEGRATION.with(|current| current.borrow().as_ref().map(f))
    }

    /// Integration point that exposes Wayland handles and lifecycle hooks.
    #[derive(Clone)]
    pub struct WaylandIntegration {
        /// Pointer to the parent Wayland surface.
        pub surface: *mut std::ffi::c_void,
        /// Pointer to the Wayland display connection.
        pub display: *mut std::ffi::c_void,
        /// Callbacks triggered before the parent surface commits.
        pre_commit_hooks: Arc<Mutex<Vec<Box<dyn Fn() + Send + Sync>>>>,
    }

    impl std::fmt::Debug for WaylandIntegration {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("WaylandIntegration")
                .field("surface", &self.surface)
                .field("display", &self.display)
                .field("pre_commit_hooks", &"<callbacks>")
                .finish()
        }
    }

    impl WaylandIntegration {
        /// Extracts the Wayland handles from a winit window.
        pub fn from_window(window: &winit::window::Window) -> Option<Self> {
            let window_handle = window.window_handle().ok()?;
            let display_handle = window.display_handle().ok()?;

            match (window_handle.as_raw(), display_handle.as_raw()) {
                (
                    RawWindowHandle::Wayland(wl_window),
                    RawDisplayHandle::Wayland(wl_display),
                ) => Some(Self {
                    surface: wl_window.surface.as_ptr(),
                    display: wl_display.display.as_ptr(),
                    pre_commit_hooks: Arc::new(Mutex::new(Vec::new())),
                }),
                _ => None,
            }
        }

        /// Registers a callback executed before the parent surface commits.
        pub fn register_pre_commit_hook(
            &self,
            hook: impl Fn() + Send + Sync + 'static,
        ) {
            if let Ok(mut hooks) = self.pre_commit_hooks.lock() {
                hooks.push(Box::new(hook));
            }
        }

        /// Triggers all registered pre-commit callbacks.
        pub fn trigger_pre_commit_hooks(&self) {
            if let Ok(hooks) = self.pre_commit_hooks.lock() {
                for hook in hooks.iter() {
                    hook();
                }
            }
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "wayland-hack")))]
/// Stubbed Wayland integration used when Wayland support is not available.
pub mod wayland {
    /// Placeholder integration returned on platforms without Wayland hooks.
    #[derive(Clone, Debug)]
    pub struct WaylandIntegration;

    impl WaylandIntegration {
        /// Always returns `None`, signalling that Wayland handles are unavailable.
        pub fn from_window(_window: &winit::window::Window) -> Option<Self> {
            None
        }

        /// Ignores registration attempts because no compositor callbacks exist.
        pub fn register_pre_commit_hook(
            &self,
            _hook: impl Fn() + Send + Sync + 'static,
        ) {
        }

        /// No-ops because there are no registered hooks on this platform.
        pub fn trigger_pre_commit_hooks(&self) {}
    }

    /// Sets the current integration to the provided value; always discarded.
    pub fn set_current_wayland_integration(
        _integration: Option<WaylandIntegration>,
    ) {
    }

    /// Runs `f` with the current integration when present; always returns `None`.
    pub fn with_current_wayland_integration<F, R>(_f: F) -> Option<R>
    where
        F: FnOnce(&WaylandIntegration) -> R,
    {
        None
    }
}
