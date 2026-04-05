#[cfg(feature = "memory-allocator-jemalloc")]
use core::ffi::c_void;

#[cfg(feature = "memory-allocator-jemalloc")]
pub(super) unsafe extern "C" fn jemalloc_malloc(size: usize) -> *mut c_void {
    tikv_jemalloc_sys::malloc(size)
}

#[cfg(feature = "memory-allocator-jemalloc")]
pub(super) unsafe extern "C" fn jemalloc_calloc(count: usize, size: usize) -> *mut c_void {
    tikv_jemalloc_sys::calloc(count, size)
}

#[cfg(feature = "memory-allocator-jemalloc")]
pub(super) unsafe extern "C" fn jemalloc_realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    tikv_jemalloc_sys::realloc(ptr, new_size)
}

#[cfg(feature = "memory-allocator-jemalloc")]
pub(super) unsafe extern "C" fn jemalloc_free(ptr: *mut c_void) {
    tikv_jemalloc_sys::free(ptr)
}
