![test](https://github.com/code-sam/graphblas_sparse_linear_algebra/actions/workflows/test_main.yml/badge.svg?branch=main)
# graphblas_sparse_linear_algebra
Workspace for a Rust wrapper for SuiteSparse:GraphBLAS. The workspace contains three crates:
- [graphblas_sparse_linear_algebra](https://crates.io/crates/graphblas_sparse_linear_algebra): the GraphBLAS API wrapper for most users
- [suitesparse_graphblas_sys](https://crates.io/crates/suitesparse_graphblas_sys): bindings to the SuiteSparse GraphBLAS implementation
- [graphblas_sparse_linear_algebra_proc_macros](https://crates.io/crates/graphblas_sparse_linear_algebra_proc_macros/): procedural macros, not for public use

## Dependencies
graphblas_sparse_linear_algebra uses the [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS) GraphBLAS implementation developed by Timothy A. Davis.

## Contributing
Awesome, contributions are welcome. Graphblas_sparse_linear_algebra and your contribution may be relicensed and integrated into commercial software in the future. Therefore, you will be asked to agree to the [Contributor License Agreement](https://github.com/code-sam/graphblas_sparse_linear_algebra/blob/main/Contributor_License_Agreement.md) when you make a pull request.

## Licensing
graphblas_sparse_linear_algebra is licensed under [Creative Commons Attribution Non Commercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode). For other licensing options, please contact Sam Dekker.
