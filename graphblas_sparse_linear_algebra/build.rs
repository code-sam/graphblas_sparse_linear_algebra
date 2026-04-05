fn main() {
    let mimalloc = std::env::var("CARGO_FEATURE_MEMORY_ALLOCATOR_MIMALLOC").is_ok();
    let jemalloc = std::env::var("CARGO_FEATURE_MEMORY_ALLOCATOR_JEMALLOC").is_ok();

    if mimalloc && jemalloc {
        panic!(
            "Features `memory-allocator-mimalloc` and `memory-allocator-jemalloc` are mutually \
             exclusive. Enable only one allocator feature at a time."
        );
    }
}
