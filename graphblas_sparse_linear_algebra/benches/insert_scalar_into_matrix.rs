use std::sync::Arc;

// extern crate graphblas_sparse_linear_algebra;

use graphblas_sparse_linear_algebra::collections::sparse_matrix::operations::FromMatrixElementList;
use graphblas_sparse_linear_algebra::context::Context;

// use graphblas_sparse_linear_algebra::operators::BinaryOperatorType;
use graphblas_sparse_linear_algebra::collections::sparse_matrix::SparseMatrix;
use graphblas_sparse_linear_algebra::operators::options::OperatorOptions;

use graphblas_sparse_linear_algebra::index::{ElementIndex, ElementIndexSelector};

use graphblas_sparse_linear_algebra::collections::sparse_matrix::{MatrixElementList, Size};
use graphblas_sparse_linear_algebra::operators::binary_operator::{Assignment, First};
use graphblas_sparse_linear_algebra::operators::insert::{
    InsertScalarIntoMatrix, InsertScalarIntoMatrixOperator,
};

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_insert_scalar_into_matrix(c: &mut Criterion) {
    let context = Context::init_default().unwrap();

    c.bench_function("test_insert_scalar_into_matrix", |b| {
        b.iter(|| bench_test(context.to_owned()))
    });
}

criterion_group!(benches, bench_insert_scalar_into_matrix);
criterion_main!(benches);

fn bench_test(context: Arc<Context>) {
    let element_list = MatrixElementList::<u8>::from_element_vector(vec![
        (1, 1, 1).into(),
        (2, 2, 2).into(),
        (2, 4, 10).into(),
        // (2, 5, 11).into(),
    ]);

    let matrix_size: Size = (10, 15).into();
    let mut matrix = SparseMatrix::<u8>::from_element_list(
        &context,
        &matrix_size,
        &element_list,
        &First::<u8>::new(),
    )
    .unwrap();

    let mask_element_list = MatrixElementList::<bool>::from_element_vector(vec![
        // (1, 1, true).into(),
        (2, 2, true).into(),
        (2, 4, true).into(),
        (2, 5, true).into(),
    ]);
    let mask = SparseMatrix::<bool>::from_element_list(
        &context,
        &matrix_size,
        &mask_element_list,
        &First::<bool>::new(),
    )
    .unwrap();

    let rows_to_insert: Vec<ElementIndex> = (0..3).collect();
    let rows_to_insert = ElementIndexSelector::Index(&rows_to_insert);
    let columns_to_insert: Vec<ElementIndex> = (0..6).collect();
    let columns_to_insert = ElementIndexSelector::Index(&columns_to_insert);

    let insert_operator = InsertScalarIntoMatrixOperator::new();

    for scalar_to_insert in 1..10 as u8 {
        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &scalar_to_insert,
                &Assignment::<u8>::new(),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &scalar_to_insert,
                &Assignment::<u8>::new(),
                &mask.to_owned(),
                &OperatorOptions::new_default(),
            )
            .unwrap();
    }
}
