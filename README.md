# graphblas_sparse_linear_algebra
Rust wrapper for SuiteSparse:GraphBLAS

## Dependencies
graphblas_sparse_linear_algebra uses the SuiteSparse:GraphBLAS GraphBLAS implementation developed by Timothy A. Davis.

By default, graphblas_sparse_linear_algebra makes a new build of SuiteSparse:GraphBLAS and links to it as a static dependency. For the source code and custom build instructions for SuiteSparse:GraphBLAS, see: https://github.com/DrTimothyAldenDavis/GraphBLAS.

## Building from source
To build graphblas_sparse_linear_algebra from source, Cmake and a C compiler must be installed. 

The git repository uses submodules, and can be cloned using:
```git clone --recurse-submodules https://github.com/code-sam/graphblas_sparse_linear_algebra.git```

## Compatibility
graphblas_sparse_linear_algebra is mostly compatible with the GraphBLAS specification version 1.3 and uses SuiteSparse:GraphBLAS v4.0.3.

## Contributing
Awesome, contributions are welcome. Graphblas_sparse_linear_algebra and your contribution may be relicensed and integrated into commercial software in the future. Therefore, you will be asked to agree to the [Contributor License Agreement](https://github.com/code-sam/graphblas_sparse_linear_algebra/blob/main/Contributor_License_Agreement.md) when you make a pull request.

## Licensing
graphblas_sparse_linear_algebra is licensed under AGPL-3.0. For other licensing options, please contact Sam Dekker.