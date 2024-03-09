use suitesparse_graphblas_sys::{
    GrB_Matrix_select_BOOL, GrB_Matrix_select_FP32, GrB_Matrix_select_FP64,
    GrB_Matrix_select_INT16, GrB_Matrix_select_INT32, GrB_Matrix_select_INT64,
    GrB_Matrix_select_INT8, GrB_Matrix_select_UINT16, GrB_Matrix_select_UINT32,
    GrB_Matrix_select_UINT64, GrB_Matrix_select_UINT8,
};

use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::operators::mask::MatrixMask;
use crate::operators::options::{GetGraphblasDescriptor, GetOptionsForOperatorWithMatrixArgument};

use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::{ConvertScalar, ValueType};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for MatrixSelector {}
unsafe impl Sync for MatrixSelector {}

#[derive(Debug, Clone)]
pub struct MatrixSelector {}

impl MatrixSelector {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait SelectFromMatrix<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        selector: &impl IndexUnaryOperator<EvaluationDomain>,
        selector_argument: &EvaluationDomain,
        argument: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArgument,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_select_from_matrix {
    ($selector_argument_type:ty, $_graphblas_implementation_type:ty, $graphblas_operator:ident) => {
        impl SelectFromMatrix<$selector_argument_type> for MatrixSelector {
            fn apply(
                &self,
                selector: &impl IndexUnaryOperator<$selector_argument_type>,
                selector_argument: &$selector_argument_type,
                argument: &impl GetGraphblasSparseMatrix,
                accumulator: &impl AccumulatorBinaryOperator<$selector_argument_type>,
                product: &mut impl GetGraphblasSparseMatrix,
                mask: &impl MatrixMask,
                options: &impl GetOptionsForOperatorWithMatrixArgument,
            ) -> Result<(), SparseLinearAlgebraError> {
                let selector_argument = selector_argument.to_owned().to_type()?;
                argument.context_ref().call(
                    || unsafe {
                        $graphblas_operator(
                            product.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            selector.graphblas_type(),
                            argument.graphblas_matrix(),
                            selector_argument,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { product.graphblas_matrix_ref() },
                )?;

                Ok(())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_select_from_matrix,
    GrB_Matrix_select
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementValue,
    };
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First};

    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::operators::index_unary_operator::{
        IsOnDiagonal, IsOnOrAboveDiagonal, IsOnOrBelowDiagonal, IsValueGreaterThan, IsValueLessThan,
    };
    use crate::operators::mask::SelectEntireMatrix;
    use crate::operators::options::OptionsForOperatorWithMatrixArgument;

    #[test]
    fn test_lower_triangle() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let index_operator = IsOnOrBelowDiagonal::new();
        let selector = MatrixSelector::new();

        let diagonal_index = 0;

        selector
            .apply(
                &index_operator,
                &diagonal_index,
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_matrix.element_value_or_default(&0, &0).unwrap(), 1);
        assert_eq!(product_matrix.element_value_or_default(&1, &0).unwrap(), 2);
        assert_eq!(product_matrix.element_value(&0, &1).unwrap(), None);
        assert_eq!(product_matrix.element_value_or_default(&1, &1).unwrap(), 4);

        let diagonal_index = -1;

        selector
            .apply(
                &index_operator,
                &diagonal_index,
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 1);
        assert_eq!(product_matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(product_matrix.element_value_or_default(&1, &0).unwrap(), 2);
        assert_eq!(product_matrix.element_value(&0, &1).unwrap(), None);
        assert_eq!(product_matrix.element_value(&1, &1).unwrap(), None);
    }

    #[test]
    fn test_upper_triangle() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let index_operator = IsOnOrAboveDiagonal::new();
        let selector = MatrixSelector::new();

        let diagonal_index = 0;

        selector
            .apply(
                &index_operator,
                &diagonal_index,
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_matrix.element_value_or_default(&0, &0).unwrap(), 1);
        assert_eq!(product_matrix.element_value(&1, &0).unwrap(), None);
        assert_eq!(product_matrix.element_value_or_default(&0, &1).unwrap(), 3);
        assert_eq!(product_matrix.element_value_or_default(&1, &1).unwrap(), 4);

        let diagonal_index = -1;

        selector
            .apply(
                &index_operator,
                &diagonal_index,
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.element_value_or_default(&0, &0).unwrap(), 1);
        assert_eq!(product_matrix.element_value_or_default(&1, &0).unwrap(), 2);
        assert_eq!(product_matrix.element_value_or_default(&0, &1).unwrap(), 3);
        assert_eq!(product_matrix.element_value_or_default(&1, &1).unwrap(), 4);
    }

    #[test]
    fn test_diagonal() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let index_operator = IsOnDiagonal::new();
        let selector = MatrixSelector::new();

        let diagonal_index = 0;

        selector
            .apply(
                &index_operator,
                &diagonal_index,
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 2);
        assert_eq!(product_matrix.element_value_or_default(&0, &0).unwrap(), 1);
        assert_eq!(product_matrix.element_value(&1, &0).unwrap(), None);
        assert_eq!(product_matrix.element_value(&0, &1).unwrap(), None);
        assert_eq!(product_matrix.element_value_or_default(&1, &1).unwrap(), 4);

        let diagonal_index = -1;
        // let diagonal_index = DiagonalIndex::Default();

        selector
            .apply(
                &index_operator,
                &diagonal_index,
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 1);
        assert_eq!(product_matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(product_matrix.element_value_or_default(&1, &0).unwrap(), 2);
        assert_eq!(product_matrix.element_value_or_default(&0, &1).unwrap(), 0);
        assert_eq!(product_matrix.element_value(&1, &1).unwrap(), None);
    }

    // #[test]
    // fn test_clear_diagonal() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();

    //     let element_list = MatrixElementList::<u8>::from_element_vector(vec![
    //         (0, 0, 1).into(),
    //         (1, 0, 2).into(),
    //         (0, 1, 3).into(),
    //         (1, 1, 4).into(),
    //     ]);

    //     let matrix_size: Size = (2, 2).into();
    //     let matrix = SparseMatrix::<u8>::from_element_list(
    //         &context.to_owned(),
    //         &matrix_size,
    //         &element_list,
    //         &First::<u8, u8, u8, u8>::new(),
    //     )
    //     .unwrap();

    //     let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

    //     let selector = MatrixSelector::new(&OperatorOptions::new_default(), None);

    //     let diagonal_index = 0;

    //     selector
    //         .clear_diagonal(&matrix, &mut product_matrix, &diagonal_index)
    //         .unwrap();

    //     println!("{}", product_matrix);

    //     assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 2);
    //     assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
    //     assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
    //     assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
    //     assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 0);

    //     let diagonal_index = -1;

    //     selector
    //         .clear_diagonal(&matrix, &mut product_matrix, &diagonal_index)
    //         .unwrap();

    //     println!("{}", product_matrix);

    //     assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
    //     assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
    //     assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 0);
    //     assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
    //     assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);
    // }

    #[test]
    fn test_zero_selector() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let index_operator = IsValueGreaterThan::<u8>::new();
        let selector = MatrixSelector::new();

        selector
            .apply(
                &index_operator,
                &0,
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.element_value_or_default(&0, &0).unwrap(), 1);
        assert_eq!(product_matrix.element_value_or_default(&1, &0).unwrap(), 2);
        assert_eq!(product_matrix.element_value_or_default(&0, &1).unwrap(), 3);
        assert_eq!(product_matrix.element_value_or_default(&1, &1).unwrap(), 4);

        let index_operator = IsValueLessThan::<u8>::new();
        let selector = MatrixSelector::new();
        selector
            .apply(
                &index_operator,
                &0,
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 0);
        assert_eq!(product_matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(product_matrix.element_value(&1, &0).unwrap(), None);
        assert_eq!(product_matrix.element_value(&0, &1).unwrap(), None);
        assert_eq!(product_matrix.element_value(&1, &1).unwrap(), None);
    }

    // #[test]
    // fn test_scalar_selector() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();

    //     let element_list = MatrixElementList::<u8>::from_element_vector(vec![
    //         (0, 0, 1).into(),
    //         (1, 0, 2).into(),
    //         (0, 1, 3).into(),
    //         (1, 1, 4).into(),
    //     ]);

    //     let matrix_size: Size = (2, 2).into();
    //     let matrix = SparseMatrix::<u8>::from_element_list(
    //         &context.to_owned(),
    //         &matrix_size,
    //         &element_list,
    //         &First::<u8, u8, u8, u8>::new(),
    //     )
    //     .unwrap();

    //     let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

    //     let selector = MatrixSelector::new(&OperatorOptions::new_default(), None);

    //     selector
    //         .greater_than_scalar(&matrix, &mut product_matrix, &1)
    //         .unwrap();

    //     println!("{}", product_matrix);

    //     assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
    //     assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
    //     assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
    //     assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
    //     assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);

    //     selector
    //         .less_than_scalar(&matrix, &mut product_matrix, &3)
    //         .unwrap();

    //     println!("{}", product_matrix);

    //     assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 2);
    //     assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
    //     assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
    //     assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 0);
    //     assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 0);
    // }
}
