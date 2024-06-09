use suitesparse_graphblas_sys::{
    GrB_Matrix_apply_BinaryOp1st_BOOL, GrB_Matrix_apply_BinaryOp1st_FP32,
    GrB_Matrix_apply_BinaryOp1st_FP64, GrB_Matrix_apply_BinaryOp1st_INT16,
    GrB_Matrix_apply_BinaryOp1st_INT32, GrB_Matrix_apply_BinaryOp1st_INT64,
    GrB_Matrix_apply_BinaryOp1st_INT8, GrB_Matrix_apply_BinaryOp1st_UINT16,
    GrB_Matrix_apply_BinaryOp1st_UINT32, GrB_Matrix_apply_BinaryOp1st_UINT64,
    GrB_Matrix_apply_BinaryOp1st_UINT8, GrB_Matrix_apply_BinaryOp2nd_BOOL,
    GrB_Matrix_apply_BinaryOp2nd_FP32, GrB_Matrix_apply_BinaryOp2nd_FP64,
    GrB_Matrix_apply_BinaryOp2nd_INT16, GrB_Matrix_apply_BinaryOp2nd_INT32,
    GrB_Matrix_apply_BinaryOp2nd_INT64, GrB_Matrix_apply_BinaryOp2nd_INT8,
    GrB_Matrix_apply_BinaryOp2nd_UINT16, GrB_Matrix_apply_BinaryOp2nd_UINT32,
    GrB_Matrix_apply_BinaryOp2nd_UINT64, GrB_Matrix_apply_BinaryOp2nd_UINT8,
    GrB_Vector_apply_BinaryOp1st_BOOL, GrB_Vector_apply_BinaryOp1st_FP32,
    GrB_Vector_apply_BinaryOp1st_FP64, GrB_Vector_apply_BinaryOp1st_INT16,
    GrB_Vector_apply_BinaryOp1st_INT32, GrB_Vector_apply_BinaryOp1st_INT64,
    GrB_Vector_apply_BinaryOp1st_INT8, GrB_Vector_apply_BinaryOp1st_UINT16,
    GrB_Vector_apply_BinaryOp1st_UINT32, GrB_Vector_apply_BinaryOp1st_UINT64,
    GrB_Vector_apply_BinaryOp1st_UINT8, GrB_Vector_apply_BinaryOp2nd_BOOL,
    GrB_Vector_apply_BinaryOp2nd_FP32, GrB_Vector_apply_BinaryOp2nd_FP64,
    GrB_Vector_apply_BinaryOp2nd_INT16, GrB_Vector_apply_BinaryOp2nd_INT32,
    GrB_Vector_apply_BinaryOp2nd_INT64, GrB_Vector_apply_BinaryOp2nd_INT8,
    GrB_Vector_apply_BinaryOp2nd_UINT16, GrB_Vector_apply_BinaryOp2nd_UINT32,
    GrB_Vector_apply_BinaryOp2nd_UINT64, GrB_Vector_apply_BinaryOp2nd_UINT8,
};

use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::CallGraphBlasContext;
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::{MatrixMask, VectorMask};
use crate::operators::options::{
    GetOperatorOptions, GetOptionsForOperatorWithMatrixAsFirstArgument,
    GetOptionsForOperatorWithMatrixAsSecondArgument,
};

use crate::operators::binary_operator::BinaryOperator;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_4_typed_graphblas_functions_with_implementation_type;
use crate::value_type::{ConvertScalar, ValueType};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mutable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for BinaryOperatorApplier {}
unsafe impl Sync for BinaryOperatorApplier {}

#[derive(Debug, Clone)]
pub struct BinaryOperatorApplier {}

impl BinaryOperatorApplier {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyBinaryOperator<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn apply_with_vector_as_left_argument(
        &self,
        first_argument: &impl GetGraphblasSparseVector,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &EvaluationDomain,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_right_argument(
        &self,
        first_argument: &EvaluationDomain,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_left_argument(
        &self,
        first_argument: &impl GetGraphblasSparseMatrix,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &EvaluationDomain,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixAsFirstArgument,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_right_argument(
        &self,
        first_argument: &EvaluationDomain,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixAsSecondArgument,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_apply_binary_operator {
    ($value_type: ty, $_implementation_type: ty, $graphblas_function_1: ident, $graphblas_function_2: ident, $graphblas_function_3: ident, $graphblas_function_4: ident) => {
        impl ApplyBinaryOperator<$value_type> for BinaryOperatorApplier {
            fn apply_with_vector_as_left_argument(
                &self,
                first_argument: &impl GetGraphblasSparseVector,
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &$value_type,
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut impl GetGraphblasSparseVector,
                mask: &impl VectorMask,
                options: &impl GetOperatorOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let second_argument = second_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_1(
                            product.graphblas_vector(),
                            mask.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            first_argument.graphblas_vector(),
                            second_argument,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_vector_as_right_argument(
                &self,
                first_argument: &$value_type,
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &impl GetGraphblasSparseVector,
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut impl GetGraphblasSparseVector,
                mask: &impl VectorMask,
                options: &impl GetOperatorOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let first_argument = first_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_2(
                            product.graphblas_vector(),
                            mask.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            first_argument,
                            second_argument.graphblas_vector(),
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_left_argument(
                &self,
                first_argument: &impl GetGraphblasSparseMatrix,
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &$value_type,
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut impl GetGraphblasSparseMatrix,
                mask: &impl MatrixMask,
                options: &impl GetOptionsForOperatorWithMatrixAsFirstArgument,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let second_argument = second_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_3(
                            product.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            first_argument.graphblas_matrix(),
                            second_argument,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_right_argument(
                &self,
                first_argument: &$value_type,
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &impl GetGraphblasSparseMatrix,
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut impl GetGraphblasSparseMatrix,
                mask: &impl MatrixMask,
                options: &impl GetOptionsForOperatorWithMatrixAsSecondArgument,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let first_argument = first_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_4(
                            product.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            first_argument,
                            second_argument.graphblas_matrix(),
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_4_typed_graphblas_functions_with_implementation_type!(
    implement_apply_binary_operator,
    GrB_Vector_apply_BinaryOp2nd,
    GrB_Vector_apply_BinaryOp1st,
    GrB_Matrix_apply_BinaryOp2nd,
    GrB_Matrix_apply_BinaryOp1st
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementValue,
    };
    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::collections::sparse_vector::operations::{
        FromVectorElementList, GetSparseVectorElementValue,
    };
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First, Plus};
    use crate::operators::mask::{SelectEntireMatrix, SelectEntireVector};
    use crate::operators::options::{
        OperatorOptions, OptionsForOperatorWithMatrixAsFirstArgument,
        OptionsForOperatorWithMatrixAsSecondArgument,
    };

    #[test]
    fn test_matrix_binary_operator_application() {
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

        let second_argument = 10u8;
        // BinaryOperatorApplier::<u8>::apply_with_matrix_as_first_argument(&operator, &matrix, &second_argument, &mut product_matrix)
        // .unwrap();
        operator
            .apply_with_matrix_as_left_argument(
                &matrix,
                &First::<u8>::new(),
                &second_argument,
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
        let first_argument = 10;
        operator
            .apply_with_matrix_as_right_argument(
                &first_argument,
                &First::<u8>::new(),
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(context.clone()),
                &OptionsForOperatorWithMatrixAsSecondArgument::new_default(),
            )
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.element_value(&2, &1).unwrap(), Some(10));
        assert_eq!(product_matrix.element_value(&9, &1).unwrap(), None);
    }

    #[test]
    fn test_vector_binary_operator_application() {
        let context = Context::init_default().unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            context.clone(),
            vector_length,
            element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(context.clone(), vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_argument = 10;
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &First::<u8>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(context.clone()),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.element_value(&9).unwrap(), None);

        let operator = BinaryOperatorApplier::new();
        let first_argument = 10;
        operator
            .apply_with_vector_as_right_argument(
                &first_argument,
                &First::<u8>::new(),
                &vector,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(context.clone()),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 10);
        assert_eq!(product_vector.element_value(&9).unwrap(), None);
    }

    #[test]
    fn test_vector_binary_operator_application_with_usize() {
        let context = Context::init_default().unwrap();

        let element_list = VectorElementList::<usize>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<usize>::from_element_list(
            context.clone(),
            vector_length,
            element_list,
            &First::<usize>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<usize>::new(context.clone(), vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_argument = 10;
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &First::<usize>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(context.clone()),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.element_value(&9).unwrap(), None);

        let operator = BinaryOperatorApplier::new();
        let first_argument = 10;
        operator
            .apply_with_vector_as_right_argument(
                &first_argument,
                &First::<usize>::new(),
                &vector,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(context.clone()),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 10);
        assert_eq!(product_vector.element_value(&9).unwrap(), None);
    }

    #[test]
    fn test_type_casting() {
        let context = Context::init_default().unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            context.clone(),
            vector_length,
            element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<i8>::new(context.clone(), vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_argument = true;
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &Plus::<bool>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(context.clone()),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 1);
        assert_eq!(product_vector.element_value(&9).unwrap(), None);

        let operator = BinaryOperatorApplier::new();

        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &Plus::<bool>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(context.clone()),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 1);
        assert_eq!(product_vector.element_value(&9).unwrap(), None);
    }
}
