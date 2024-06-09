use suitesparse_graphblas_sys::{
    GrB_Matrix_apply_BinaryOp1st_Scalar, GrB_Matrix_apply_BinaryOp2nd_Scalar,
    GrB_Vector_apply_BinaryOp1st_Scalar, GrB_Vector_apply_BinaryOp2nd_Scalar,
};

use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_scalar::GraphblasSparseScalarTrait;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::CallGraphBlasContext;
use crate::error::SparseLinearAlgebraError;
use crate::operators::apply::BinaryOperatorApplier;
use crate::operators::binary_operator::{AccumulatorBinaryOperator, BinaryOperator};
use crate::operators::mask::{MatrixMask, VectorMask};
use crate::operators::options::{
    GetOperatorOptions, GetOptionsForOperatorWithMatrixAsFirstArgument,
    GetOptionsForOperatorWithMatrixAsSecondArgument,
};
use crate::value_type::ValueType;

pub trait ApplyBinaryOperatorWithSparseScalar<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn apply_with_vector_as_left_argument(
        &self,
        left_argument: &impl GetGraphblasSparseVector,
        operator: &impl BinaryOperator<EvaluationDomain>,
        right_argument: &impl GraphblasSparseScalarTrait,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_right_argument(
        &self,
        left_argument: &impl GraphblasSparseScalarTrait,
        operator: &impl BinaryOperator<EvaluationDomain>,
        right_argument: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_left_argument(
        &self,
        left_argument: &impl GetGraphblasSparseMatrix,
        operator: &impl BinaryOperator<EvaluationDomain>,
        right_argument: &impl GraphblasSparseScalarTrait,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixAsFirstArgument,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_right_argument(
        &self,
        left_argument: &impl GraphblasSparseScalarTrait,
        operator: &impl BinaryOperator<EvaluationDomain>,
        rigth_argument: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixAsSecondArgument,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyBinaryOperatorWithSparseScalar<EvaluationDomain>
    for BinaryOperatorApplier
{
    fn apply_with_vector_as_left_argument(
        &self,
        first_argument: &impl GetGraphblasSparseVector,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &impl GraphblasSparseScalarTrait,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Vector_apply_BinaryOp2nd_Scalar(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    first_argument.graphblas_vector(),
                    second_argument.graphblas_scalar(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_with_vector_as_right_argument(
        &self,
        first_argument: &impl GraphblasSparseScalarTrait,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Vector_apply_BinaryOp1st_Scalar(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    first_argument.graphblas_scalar(),
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
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &impl GraphblasSparseScalarTrait,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixAsFirstArgument,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_BinaryOp2nd_Scalar(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    first_argument.graphblas_matrix(),
                    second_argument.graphblas_scalar(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;

        Ok(())
    }

    fn apply_with_matrix_as_right_argument(
        &self,
        first_argument: &impl GraphblasSparseScalarTrait,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixAsSecondArgument,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_BinaryOp1st_Scalar(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    first_argument.graphblas_scalar(),
                    second_argument.graphblas_matrix(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementValue,
    };
    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::collections::sparse_scalar::SparseScalar;
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

        let second_argument = SparseScalar::<u8>::from_value(context.clone(), 10).unwrap();
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
        let first_argument = SparseScalar::<u8>::from_value(context.clone(), 10).unwrap();
        operator
            .apply_with_matrix_as_right_argument(
                &first_argument,
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

        let second_argument = SparseScalar::<u8>::from_value(context.clone(), 10).unwrap();
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
        let first_argument = SparseScalar::<u8>::from_value(context.clone(), 10).unwrap();
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

        let second_argument = SparseScalar::<usize>::from_value(context.clone(), 10).unwrap();
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
        let first_argument = SparseScalar::<usize>::from_value(context.clone(), 10).unwrap();
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

        let second_argument = SparseScalar::<bool>::from_value(context.clone(), true).unwrap();
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &Plus::<u8>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(context.clone()),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 3);
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
