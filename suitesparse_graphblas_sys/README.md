# suitesparse_graphblas_sys
Rust bindings to SuiteSparse:GraphBLAS. Crates should not directly use these bindings. Instead, the crate [graphblas_sparse_linear_algebra](https://crates.io/crates/graphblas_sparse_linear_algebra) provides a safe and idiomatic wrapper.

## GraphBLAS implementation
suitesparse_graphblas_sys wraps the [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS) v8.2.0. GraphBLAS implementation from Timothy A. Davis. This implementation is mostly compatible with the GraphBLAS specification version 2.0.

## Requirements
suitesparse_graphblas_sys uses the SuiteSparse:GraphBLAS GraphBLAS implementation developed by Timothy A. Davis.

By default, graphblas_sparse_linear_algebra makes a new build of SuiteSparse:GraphBLAS and links to it as a static dependency.

To install the required packages listed below on Ubuntu 24.04:

```
sudo apt install git build-essential pkg-config libssl-dev cmake clang
```

### Git
suitesparse_graphblas_sys clones the [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS) repository upon build. The build system must have Git installed and an internet connection.

### C compiler
suitesparse_graphblas_sys includes a build script to build SuiteSparse:GraphBLAS from source. The build process requires an installed C compiler, for example [GCC](https://gcc.gnu.org/).

### Environment variables
suitesparse_graphblas_sys' build script reads the following environment variables:
- SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH with the path of the C compiler on the build system. For example, when using GCC11 on Ubuntu 22.04 please set the environment variable to: "/usr/lib/gcc/x86_64-linux-gnu/11"
- SUITESPARSE_GRAPHBLAS_SYS_OPENMP_STATIC_LIBRARY_NAME with the name of the [openmp](https://www.openmp.org/) implementation to be used. For example, when using GCC please set the environment variable to "gomp"

### Bindgen and CMake
suitesparse_graphblas_sys depends on bindgen, please make sure the build system meets its [requirements](https://rust-lang.github.io/rust-bindgen/requirements.html)

suitesparse_graphblas_sys also requires [CMake](https://cmake.org/) installed on the build system.

## Contributing
Awesome, contributions are welcome. suitesparse_graphblas_sys and your contribution may be relicensed and integrated into commercial software in the future. Therefore, you will be asked to agree to the [Contributor License Agreement](https://github.com/code-sam/graphblas_sparse_linear_algebra/blob/main/Contributor_License_Agreement.md) when you make a pull request.

## Licensing
suitesparse_graphblas_sys is licensed under [Creative Commons Attribution Non Commercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode). For other licensing options, please contact Sam Dekker.

## Compatibility
Tested on Ubuntu 22.04.1 LTS with the distribution standard GCC compiler.
