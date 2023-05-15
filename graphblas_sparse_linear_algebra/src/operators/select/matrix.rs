use std::ptr;

use std::marker::PhantomData;

use suitesparse_graphblas_sys::{
    GrB_IndexUnaryOp, GrB_Matrix_select_BOOL, GrB_Matrix_select_FP32, GrB_Matrix_select_FP64,
    GrB_Matrix_select_INT16, GrB_Matrix_select_INT32, GrB_Matrix_select_INT64,
    GrB_Matrix_select_INT8, GrB_Matrix_select_UINT16, GrB_Matrix_select_UINT32,
    GrB_Matrix_select_UINT64, GrB_Matrix_select_UINT8,
};

use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::OperatorOptions;

use crate::collections::sparse_matrix::GraphblasSparseMatrixTrait;
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::{ConvertScalar, ValueType};

use crate::bindings_to_graphblas_implementation::{GrB_BinaryOp, GrB_Descriptor};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<EvaluationDomain: ValueType> Send for MatrixSelector<EvaluationDomain> {}
unsafe impl<EvaluationDomain: ValueType> Sync for MatrixSelector<EvaluationDomain> {}

#[derive(Debug, Clone)]
pub struct MatrixSelector<EvaluationDomain: ValueType> {
    _evaluation_domain: PhantomData<EvaluationDomain>,

    selector: GrB_IndexUnaryOp,
    accumulator: GrB_BinaryOp,
    options: GrB_Descriptor,
}

impl<EvaluationDomain: ValueType> MatrixSelector<EvaluationDomain> {
    pub fn new(
        selector: &impl IndexUnaryOperator<EvaluationDomain>,
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
    ) -> Self {
        Self {
            selector: selector.graphblas_type(),
            accumulator: accumulator.accumulator_graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _evaluation_domain: PhantomData,
        }
    }
}

pub trait SelectFromMatrix<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        selector_argument: &EvaluationDomain,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_mask(
        &self,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        selector_argument: &EvaluationDomain,
        mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_select_from_matrix {
    ($selector_argument_type:ty, $_graphblas_implementation_type:ty, $graphblas_operator:ident) => {
        impl SelectFromMatrix<$selector_argument_type> for MatrixSelector<$selector_argument_type> {
            fn apply(
                &self,
                argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
                product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
                selector_argument: &$selector_argument_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let selector_argument = selector_argument.clone().to_type()?;
                argument.context_ref().call(
                    || unsafe {
                        $graphblas_operator(
                            product.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            self.selector,
                            argument.graphblas_matrix(),
                            selector_argument,
                            self.options,
                        )
                    },
                    unsafe { product.graphblas_matrix_ref() },
                )?;

                Ok(())
            }

            fn apply_with_mask(
                &self,
                argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
                product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
                selector_argument: &$selector_argument_type,
                mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
            ) -> Result<(), SparseLinearAlgebraError> {
                let selector_argument = selector_argument.clone().to_type()?;
                argument.context_ref().call(
                    || unsafe {
                        $graphblas_operator(
                            product.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            self.accumulator,
                            self.selector,
                            argument.graphblas_matrix(),
                            selector_argument,
                            self.options,
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

    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size, SparseMatrix,
    };
    use crate::operators::index_unary_operator::{
        IsOnDiagonal, IsOnOrAboveDiagonal, IsOnOrBelowDiagonal, IsValueGreaterThan, IsValueLessThan,
    };

    #[test]
    fn test_lower_triangle() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let index_operator = IsOnOrBelowDiagonal::new();
        let selector = MatrixSelector::<i8>::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        let diagonal_index = 0;

        selector
            .apply(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            1
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 0).into())
                .unwrap(),
            2
        );
        assert_eq!(
            product_matrix.get_element_value(&(0, 1).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            4
        );

        let diagonal_index = -1;

        selector
            .apply(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 1);
        assert_eq!(
            product_matrix.get_element_value(&(0, 0).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 0).into())
                .unwrap(),
            2
        );
        assert_eq!(
            product_matrix.get_element_value(&(0, 1).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix.get_element_value(&(1, 1).into()).unwrap(),
            None
        );
    }

    #[test]
    fn test_upper_triangle() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let index_operator = IsOnOrAboveDiagonal::new();
        let selector = MatrixSelector::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        let diagonal_index = 0;

        selector
            .apply(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            1
        );
        assert_eq!(
            product_matrix.get_element_value(&(1, 0).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 1).into())
                .unwrap(),
            3
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            4
        );

        let diagonal_index = -1;

        selector
            .apply(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            1
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 0).into())
                .unwrap(),
            2
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 1).into())
                .unwrap(),
            3
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            4
        );
    }

    #[test]
    fn test_diagonal() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let index_operator = IsOnDiagonal::new();
        let selector = MatrixSelector::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        let diagonal_index = 0;

        selector
            .apply(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 2);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            1
        );
        assert_eq!(
            product_matrix.get_element_value(&(1, 0).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix.get_element_value(&(0, 1).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            4
        );

        let diagonal_index = -1;
        // let diagonal_index = DiagonalIndex::Default();

        selector
            .apply(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 1);
        assert_eq!(
            product_matrix.get_element_value(&(0, 0).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 0).into())
                .unwrap(),
            2
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 1).into())
                .unwrap(),
            0
        );
        assert_eq!(
            product_matrix.get_element_value(&(1, 1).into()).unwrap(),
            None
        );
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
    //         &context.clone(),
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
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let index_operator = IsValueGreaterThan::<u8>::new();
        let selector = MatrixSelector::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        selector.apply(&matrix, &mut product_matrix, &0).unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            1
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 0).into())
                .unwrap(),
            2
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(0, 1).into())
                .unwrap(),
            3
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            4
        );

        let index_operator = IsValueLessThan::<u8>::new();
        let selector = MatrixSelector::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );
        selector.apply(&matrix, &mut product_matrix, &0).unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 0);
        assert_eq!(
            product_matrix.get_element_value(&(0, 0).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix.get_element_value(&(1, 0).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix.get_element_value(&(0, 1).into()).unwrap(),
            None
        );
        assert_eq!(
            product_matrix.get_element_value(&(1, 1).into()).unwrap(),
            None
        );
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
    //         &context.clone(),
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
