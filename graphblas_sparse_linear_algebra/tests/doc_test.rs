mod tests {
    use graphblas_sparse_linear_algebra::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementValue,
    };
    use graphblas_sparse_linear_algebra::collections::sparse_matrix::{
        MatrixElementList, Size, SparseMatrix,
    };
    use graphblas_sparse_linear_algebra::collections::Collection;
    use graphblas_sparse_linear_algebra::context::Context;
    use graphblas_sparse_linear_algebra::operators::apply::{
        ApplyBinaryOperator, BinaryOperatorApplier,
    };
    use graphblas_sparse_linear_algebra::operators::binary_operator::{Assignment, First};
    use graphblas_sparse_linear_algebra::operators::mask::SelectEntireMatrix;
    use graphblas_sparse_linear_algebra::operators::options::{
        OptionsForOperatorWithMatrixAsFirstArgument, OptionsForOperatorWithMatrixAsSecondArgument,
    };

    #[test]
    fn doc_test() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 1, 2).into(),
            (4, 2, 4).into(),
            (5, 2, 5).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            matrix_size,
            element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(context.clone(), matrix_size).unwrap();

        let operator = BinaryOperatorApplier::new();
        let first_argument = 10;
        operator
            .apply_with_matrix_as_left_argument(
                &matrix,
                &First::<u8>::new(),
                first_argument,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(context.clone()),
                &OptionsForOperatorWithMatrixAsFirstArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.element_value(&2, &1).unwrap(), Some(2));
        assert_eq!(product_matrix.element_value(&9, &1).unwrap(), None);

        let operator = BinaryOperatorApplier::new();
        let second_argument = 10;
        operator
            .apply_with_matrix_as_right_argument(
                second_argument,
                &First::<u8>::new(),
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(context),
                &OptionsForOperatorWithMatrixAsSecondArgument::new_default(),
            )
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.element_value(&2, &1).unwrap(), Some(10));
        assert_eq!(product_matrix.element_value(&9, &1).unwrap(), None);
    }
}
