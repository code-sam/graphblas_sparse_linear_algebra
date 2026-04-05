#[cfg(feature = "memory-allocator-mimalloc")]
use std::ffi::c_void;

#[cfg(feature = "memory-allocator-mimalloc")]
pub(super) unsafe extern "C" fn mimalloc_malloc(size: usize) -> *mut c_void {
    libmimalloc_sys::mi_malloc(size)
}

#[cfg(feature = "memory-allocator-mimalloc")]
pub(super) unsafe extern "C" fn mimalloc_calloc(count: usize, size: usize) -> *mut c_void {
    libmimalloc_sys::mi_zalloc(count.saturating_mul(size))
}

#[cfg(feature = "memory-allocator-mimalloc")]
pub(super) unsafe extern "C" fn mimalloc_realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    libmimalloc_sys::mi_realloc(ptr, new_size)
}

#[cfg(feature = "memory-allocator-mimalloc")]
pub(super) unsafe extern "C" fn mimalloc_free(ptr: *mut c_void) {
    libmimalloc_sys::mi_free(ptr)
}
