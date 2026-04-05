use std::ffi::c_void;

#[cfg(feature = "memory-allocator-jemalloc")]
use super::jemalloc::{jemalloc_calloc, jemalloc_free, jemalloc_malloc, jemalloc_realloc};

#[cfg(feature = "memory-allocator-mimalloc")]
use super::mimalloc::{mimalloc_calloc, mimalloc_free, mimalloc_malloc, mimalloc_realloc};

/// The allocator that GraphBLAS will use internally.
///
/// Must match whatever is set as the Rust global allocator, otherwise
/// memory freed by one allocator will have been allocated by another —
/// which is undefined behaviour.
pub enum MemoryAllocator {
    SystemDefault,

    /// Use mimalloc. Requires the `mimalloc` feature.
    /// The caller must also set `mimalloc::MiMalloc` as the global allocator.
    #[cfg(feature = "memory-allocator-mimalloc")]
    MiMalloc,

    /// Use jemalloc. Requires the `jemalloc` feature.
    /// The caller must also set `tikv_jemallocator::Jemalloc` as the global allocator.
    #[cfg(feature = "memory-allocator-jemalloc")]
    Jemalloc,

    /// Supply raw allocator function pointers. You are responsible for
    /// ensuring these match the active global allocator.
    Custom {
        malloc: unsafe extern "C" fn(usize) -> *mut c_void,
        calloc: unsafe extern "C" fn(usize, usize) -> *mut c_void,
        realloc: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
        free: unsafe extern "C" fn(*mut c_void),
    },
}

pub(crate) struct MemoryAllocatorFuctionPointers {
    pub malloc: unsafe extern "C" fn(usize) -> *mut c_void,
    pub calloc: unsafe extern "C" fn(usize, usize) -> *mut c_void,
    pub realloc: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
    pub free: unsafe extern "C" fn(*mut c_void),
}

impl MemoryAllocator {
    pub(crate) fn memory_allocator_function_pointers(
        &self,
    ) -> Option<MemoryAllocatorFuctionPointers> {
        match self {
            MemoryAllocator::SystemDefault => None,

            #[cfg(feature = "memory-allocator-mimalloc")]
            MemoryAllocator::MiMalloc => Some(MemoryAllocatorFuctionPointers {
                malloc: mimalloc_malloc,
                calloc: mimalloc_calloc,
                realloc: mimalloc_realloc,
                free: mimalloc_free,
            }),

            #[cfg(feature = "memory-allocator-jemalloc")]
            MemoryAllocator::Jemalloc => Some(MemoryAllocatorFuctionPointers {
                malloc: jemalloc_malloc,
                calloc: jemalloc_calloc,
                realloc: jemalloc_realloc,
                free: jemalloc_free,
            }),

            MemoryAllocator::Custom {
                malloc,
                calloc,
                realloc,
                free,
            } => Some(MemoryAllocatorFuctionPointers {
                malloc: *malloc,
                calloc: *calloc,
                realloc: *realloc,
                free: *free,
            }),
        }
    }
}
