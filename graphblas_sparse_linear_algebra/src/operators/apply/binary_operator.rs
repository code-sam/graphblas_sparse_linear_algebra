use std::ptr;

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

use crate::collections::sparse_matrix::GraphblasSparseMatrixTrait;
use crate::collections::sparse_vector::GraphblasSparseVectorTrait;
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::OperatorOptionsTrait;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
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
        left_argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
        operator: &impl BinaryOperator<EvaluationDomain>,
        right_argument: &EvaluationDomain,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_right_argument(
        &self,
        first_argument: &EvaluationDomain,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_left_argument_and_mask(
        &self,
        first_argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &EvaluationDomain,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_right_argument_and_mask(
        &self,
        first_argument: &EvaluationDomain,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_left_argument(
        &self,
        first_argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &EvaluationDomain,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_right_argument(
        &self,
        first_argument: &EvaluationDomain,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_left_argument_and_mask(
        &self,
        first_argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &EvaluationDomain,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_right_argument_and_mask(
        &self,
        first_argument: &EvaluationDomain,
        operator: &impl BinaryOperator<EvaluationDomain>,
        second_argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_apply_binary_operator {
    ($value_type: ty, $_implementation_type: ty, $graphblas_function_1: ident, $graphblas_function_2: ident, $graphblas_function_3: ident, $graphblas_function_4: ident) => {
        impl ApplyBinaryOperator<$value_type> for BinaryOperatorApplier {
            fn apply_with_vector_as_left_argument(
                &self,
                left_argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
                operator: &impl BinaryOperator<$value_type>,
                right_argument: &$value_type,
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
                options: &OperatorOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let right_argument = right_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_1(
                            product.graphblas_vector(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            left_argument.graphblas_vector(),
                            right_argument,
                            options.to_graphblas_descriptor(),
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
                second_argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
                options: &OperatorOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let first_argument = first_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_2(
                            product.graphblas_vector(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            first_argument,
                            second_argument.graphblas_vector(),
                            options.to_graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_vector_as_left_argument_and_mask(
                &self,
                first_argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &$value_type,
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
                mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
                options: &OperatorOptions,
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
                            options.to_graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_vector_as_right_argument_and_mask(
                &self,
                first_argument: &$value_type,
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
                mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
                options: &OperatorOptions,
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
                            options.to_graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_left_argument(
                &self,
                first_argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &$value_type,
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
                options: &OperatorOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let second_argument = second_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_3(
                            product.graphblas_matrix(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            first_argument.graphblas_matrix(),
                            second_argument,
                            options.to_graphblas_descriptor(),
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
                second_argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
                options: &OperatorOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let first_argument = first_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_4(
                            product.graphblas_matrix(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            first_argument,
                            second_argument.graphblas_matrix(),
                            options.to_graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_left_argument_and_mask(
                &self,
                first_argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &$value_type,
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
                mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
                options: &OperatorOptions,
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
                            options.to_graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_right_argument_and_mask(
                &self,
                first_argument: &$value_type,
                operator: &impl BinaryOperator<$value_type>,
                second_argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
                mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
                options: &OperatorOptions,
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
                            options.to_graphblas_descriptor(),
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

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size, SparseMatrix,
    };
    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, SparseVector, VectorElementList,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First, Plus};

    #[test]
    fn test_matrix_binary_operator_application() {
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
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_agrument = 10u8;
        // BinaryOperatorApplier::<u8>::apply_with_matrix_as_first_argument(&operator, &matrix, &second_agrument, &mut product_matrix)
        // .unwrap();
        operator
            .apply_with_matrix_as_left_argument(
                &matrix,
                &First::<u8>::new(),
                &second_agrument,
                &Assignment::new(),
                &mut product_matrix,
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

        let operator = BinaryOperatorApplier::new();
        let first_agrument = 10;
        operator
            .apply_with_matrix_as_right_argument(
                &first_agrument,
                &First::<u8>::new(),
                &matrix,
                &Assignment::new(),
                &mut product_matrix,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix.get_element_value(&(2, 1).into()).unwrap(),
            Some(10)
        );
        assert_eq!(
            product_matrix.get_element_value(&(9, 1).into()).unwrap(),
            None
        );
    }

    #[test]
    fn test_vector_binary_operator_application() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_agrument = 10;
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &First::<u8>::new(),
                &second_agrument,
                &Assignment::new(),
                &mut product_vector,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);

        let operator = BinaryOperatorApplier::new();
        let first_argument = 10;
        operator
            .apply_with_vector_as_right_argument(
                &first_argument,
                &First::<u8>::new(),
                &vector,
                &Assignment::new(),
                &mut product_vector,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 10);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);
    }

    #[test]
    fn test_vector_binary_operator_application_with_usize() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<usize>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<usize>::from_element_list(
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<usize>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<usize>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_agrument = 10;
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &First::<usize>::new(),
                &second_agrument,
                &Assignment::new(),
                &mut product_vector,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);

        let operator = BinaryOperatorApplier::new();
        let first_agrument = 10;
        operator
            .apply_with_vector_as_right_argument(
                &first_agrument,
                &First::<usize>::new(),
                &vector,
                &Assignment::new(),
                &mut product_vector,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 10);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);
    }

    #[test]
    fn test_type_casting() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<i8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_argument = true;
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &Plus::<bool>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 1);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);

        let operator = BinaryOperatorApplier::new();

        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &Plus::<bool>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 1);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);

        // let operator = BinaryOperatorApplier::new(
        //     &First::<u8, u8, u8, u8>::new(),
        //     &OperatorOptions::new_default(),
        //     None,
        // );
        // let first_argument = SparseScalar::<u8>::from_value(&context, 10).unwrap();
        // operator
        //     .apply_with_vector_as_second_argument(&first_argument, &vector, &mut product_vector)
        //     .unwrap();

        // println!("{}", vector);
        // println!("{}", product_vector);

        // assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        // assert_eq!(product_vector.get_element_value(&2).unwrap(), 10);
        // assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    }

    // #[test]
    // fn test_operator_destructor() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();
    //     // Test if this causes a memory leak, due to the absence of an explicit call to GrB_free.
    //     for i in 0..(1e5 as usize) {
    //         let element_list = VectorElementList::<usize>::from_element_vector(vec![
    //             (1, 1).into(),
    //             (2, 2).into(),
    //             (4, 4).into(),
    //             (5, 5).into(),
    //             (10+i, i).into(),
    //         ]);

    //         let vector_length: usize = 100+i;
    //         let vector = SparseVector::<usize>::from_element_list(
    //             &context.clone(),
    //             &vector_length,
    //             &element_list,
    //             &First::<usize, usize, usize>::new(),
    //         )
    //         .unwrap();
    //         let mut product_vector = SparseVector::<usize>::new(&context, &vector_length).unwrap();

    //         let operator = BinaryOperatorApplier::new(
    //             &Plus::<usize, usize, usize>::new(),
    //             &OperatorOptions::new_default(),
    //             None,
    //         );

    //         operator
    //             .apply_with_vector_as_second_argument(&10, &vector, &mut product_vector)
    //             .unwrap();

    //         assert_eq!(product_vector.number_of_stored_elements().unwrap(), 5);
    //         assert_eq!(product_vector.get_element_value(&(10+i)).unwrap(), i+10);
    //     }
    // }
}
