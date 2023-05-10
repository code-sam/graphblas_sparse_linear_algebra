// #[cfg(test)]
mod tests {
    use rayon::prelude::*;
    use std::sync::Mutex;

    use graphblas_sparse_linear_algebra::collections::sparse_matrix::{
        Coordinate, GetMatrixElement, MatrixElement, SetMatrixElement, Size, SparseMatrix,
    };
    use graphblas_sparse_linear_algebra::context::{Context, Mode};
    use graphblas_sparse_linear_algebra::operators::binary_operator::Plus;
    use graphblas_sparse_linear_algebra::operators::element_wise_multiplication::{
        ApplyElementWiseMatrixMultiplicationBinaryOperator,
        ElementWiseMatrixMultiplicationBinaryOperator,
    };
    use graphblas_sparse_linear_algebra::operators::options::OperatorOptions;

    #[test]
    fn parallel_calls_to_graphblas() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let number_of_matrices = 100;

        let matrix_size = Size::new(10, 5);
        let mut matrices: Vec<SparseMatrix<i32>> = (0..number_of_matrices)
            .into_par_iter()
            .map(|_| SparseMatrix::<i32>::new(&context, &matrix_size).unwrap())
            .collect();

        matrices.par_iter_mut().for_each(|matrix| {
            matrix
                .set_element(MatrixElement::from_triple(1, 2, 3))
                .unwrap()
        });

        let add_operator = Plus::<i32>::new();
        let options = OperatorOptions::new_default();
        let result_matrix = Mutex::new(SparseMatrix::<i32>::new(&context, &matrix_size).unwrap());

        let element_wise_matrix_add_operator =
            ElementWiseMatrixMultiplicationBinaryOperator::<i32, i32, i32, i32>::new(
                &add_operator,
                &options,
                &add_operator,
            );

        matrices.par_iter().for_each(|matrix| {
            element_wise_matrix_add_operator
                .apply(&matrix, &matrix, &mut result_matrix.lock().unwrap())
                .unwrap();
        });

        let result_matrix = result_matrix.into_inner().unwrap();

        assert_eq!(
            600,
            result_matrix
                .get_element(Coordinate::new(1, 2))
                .unwrap()
                .unwrap()
                .value()
        );
    }
}
