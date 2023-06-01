use graphblas_sparse_linear_algebra::collections::sparse_matrix::FromMatrixElementList;
use graphblas_sparse_linear_algebra::collections::sparse_matrix::GetMatrixElementValue;
use graphblas_sparse_linear_algebra::collections::sparse_matrix::MatrixElementList;
use graphblas_sparse_linear_algebra::collections::sparse_matrix::Size;
use graphblas_sparse_linear_algebra::collections::sparse_matrix::SparseMatrix;
use graphblas_sparse_linear_algebra::collections::Collection;
use graphblas_sparse_linear_algebra::context::ContextTrait;
use graphblas_sparse_linear_algebra::context::{Context, Mode};

use criterion::{criterion_group, criterion_main, Criterion};
use graphblas_sparse_linear_algebra::operators::apply::ApplyBinaryOperator;
use graphblas_sparse_linear_algebra::operators::apply::BinaryOperatorApplier;
use graphblas_sparse_linear_algebra::operators::binary_operator::Assignment;
use graphblas_sparse_linear_algebra::operators::binary_operator::Plus;
use graphblas_sparse_linear_algebra::operators::mask::SelectEntireMatrix;
use graphblas_sparse_linear_algebra::operators::options::OperatorOptions;

fn bench_vector_indices(c: &mut Criterion) {
    let context = Context::init_ready(Mode::NonBlocking).unwrap();

    let element_list = MatrixElementList::<u8>::from_element_vector(vec![
        (1, 1, 1).into(),
        (2, 1, 2).into(),
        (4, 2, 4).into(),
        (5, 2, 5).into(),
    ]);

    let matrix_size: Size = (1000, 1500).into();
    let matrix = SparseMatrix::<u8>::from_element_list(
        &context.to_owned(),
        &matrix_size,
        &element_list,
        &Plus::<u8>::new(),
    )
    .unwrap();

    let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

    c.bench_function("Apply binary operator", |b| {
        b.iter(|| bench_apply_binary_operator(&matrix, &mut product_matrix))
    });
}

criterion_group!(benches, bench_vector_indices);
criterion_main!(benches);

fn bench_apply_binary_operator(matrix: &SparseMatrix<u8>, product_matrix: &mut SparseMatrix<u8>) {
    let operator = BinaryOperatorApplier::new();

    let second_agrument = 10u8;

    operator
        .apply_with_matrix_as_left_argument(
            matrix,
            &Plus::<u8>::new(),
            &second_agrument,
            &Assignment::new(),
            product_matrix,
            &SelectEntireMatrix::new(&matrix.context_ref()),
            &OperatorOptions::new_default(),
        )
        .unwrap();

    println!("{}", product_matrix);

    assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
    assert_eq!(
        product_matrix.get_element_value(&(2, 1).into()).unwrap(),
        Some(2)
    );
    assert_eq!(
        product_matrix.get_element_value(&(9, 1).into()).unwrap(),
        None
    );
}
