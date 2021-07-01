# graphblas_sparse_linear_algebra
Rust wrapper for SuiteSparse:GraphBLAS

## Compatibility
Tested with SuiteSparse:GraphBLAS version 4.0.3. on Ubuntu 18.04.

## Dependencies
graphblas_sparse_linear_algebra uses the SuiteSparse:GraphBLAS GraphBLAS implementation developed by Timothy A. Davis.

graphblas_sparse_linear_algebra requires the following files present in the graphblas_implementation/ directory:
- graphblas_implementation/GraphBLAS.h
- graphblas_implementation/libgaphblas.so

For the source code and build instructions of SuiteSparse:GraphBLAS, see: https://github.com/DrTimothyAldenDavis/GraphBLAS

graphblas_sparse_linear_algebra also requires gomp to be installed locally in /usr/lib/gcc/x86_64-linux-gnu/7/
