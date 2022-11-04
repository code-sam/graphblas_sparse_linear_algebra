![test](https://github.com/code-sam/graphblas_sparse_linear_algebra/actions/workflows/test_main.yml/badge.svg?branch=main)
# graphblas_sparse_linear_algebra
Rust wrapper for SuiteSparse:GraphBLAS

## Minimum example
```rust
    use graphblas_sparse_linear_algebra::context::{Context, Mode};
    use graphblas_sparse_linear_algebra::operators::apply::{BinaryOperatorApplier, BinaryOperatorApplierTrait};
    use graphblas_sparse_linear_algebra::operators::binary_operator::{First};
    use graphblas_sparse_linear_algebra::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
    use graphblas_sparse_linear_algebra::value_types::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size, SparseMatrix
    };
    use graphblas_sparse_linear_algebra::value_types::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, VectorElementList,
    };

fn main() {
    let context = Context::init_ready(Mode::NonBlocking).unwrap();

    let element_list = MatrixElementList::<u8>::from_element_vector(vec![
        (1, 1, 1).into(),
        (2, 1, 2).into(),
        (4, 2, 4).into(),
        (5, 2, 5).into(),
    ]);

    let matrix_size: Size = (10, 15).into();
    let matrix = SparseMatrix::<u8>::from_element_list(
        &context.clone(),
        &matrix_size,
        &element_list,
        &First::<u8, u8, u8>::new(),
    )
    .unwrap();

    let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

    let operator = BinaryOperatorApplier::new(
        &First::<u8, u8, u8>::new(),
        &OperatorOptions::new_default(),
        None,
    );

    operator
        .apply_with_matrix_as_first_argument(&matrix, &10, &mut product_matrix)
        .unwrap();

    println!("{}", product_matrix);

    assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
    assert_eq!(product_matrix.get_element_value(&(2, 1).into()).unwrap(), 2);
    assert_eq!(product_matrix.get_element_value(&(9, 1).into()).unwrap(), 0);

    let operator = BinaryOperatorApplier::new(
        &First::<u8, u8, u8>::new(),
        &OperatorOptions::new_default(),
        None,
    );
    operator
        .apply_with_matrix_as_second_argument(&10, &matrix, &mut product_matrix)
        .unwrap();

    println!("{}", matrix);
    println!("{}", product_matrix);

    assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
    assert_eq!(
        product_matrix.get_element_value(&(2, 1).into()).unwrap(),
        10
    );
    assert_eq!(product_matrix.get_element_value(&(9, 1).into()).unwrap(), 0);
 }
 ```

## Dependencies
graphblas_sparse_linear_algebra uses the [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS) GraphBLAS implementation developed by Timothy A. Davis.

## Building from source
Please make sure to meet the requirements for building [suitesparse_graphblas_sys](https://crates.io/crates/suitesparse_graphblas_sys).

## Compatibility
graphblas_sparse_linear_algebra is mostly compatible with the GraphBLAS specification version 2.0 and uses SuiteSparse:GraphBLAS v7.3.0.

## Contributing
Awesome, contributions are welcome. Graphblas_sparse_linear_algebra and your contribution may be relicensed and integrated into commercial software in the future. Therefore, you will be asked to agree to the [Contributor License Agreement](https://github.com/code-sam/graphblas_sparse_linear_algebra/blob/main/Contributor_License_Agreement.md) when you make a pull request.

## Licensing
graphblas_sparse_linear_algebra is licensed under [Creative Commons Attribution Non Commercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode). For other licensing options, please contact Sam Dekker.
