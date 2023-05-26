use std::ptr;

use crate::collections::sparse_matrix::GraphblasSparseMatrixTrait;
use crate::collections::sparse_vector::GraphblasSparseVectorTrait;
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::OperatorOptionsTrait;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::ValueType;

use crate::bindings_to_graphblas_implementation::GrB_Matrix_reduce_BinaryOp;

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for BinaryOperatorReducer {}
unsafe impl Sync for BinaryOperatorReducer {}

#[derive(Debug, Clone)]
pub struct BinaryOperatorReducer {}

impl BinaryOperatorReducer {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ReduceWithBinaryOperator<EvaluationDomain: ValueType> {
    fn to_column_vector(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn to_colunm_vector_with_mask(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn to_row_vector(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn to_row_vector_with_mask(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ReduceWithBinaryOperator<EvaluationDomain>
    for BinaryOperatorReducer
{
    fn to_column_vector(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_reduce_BinaryOp(
                    product.graphblas_vector(),
                    ptr::null_mut(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    argument.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
    }

    fn to_colunm_vector_with_mask(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_reduce_BinaryOp(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    argument.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
    }

    fn to_row_vector(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        self.to_column_vector(
            operator,
            argument,
            accumulator,
            product,
            &options.with_negated_transpose_input0(),
        )
    }

    fn to_row_vector_with_mask(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        self.to_colunm_vector_with_mask(
            operator,
            argument,
            accumulator,
            product,
            mask,
            &options.with_negated_transpose_input0(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First, Plus};

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, MatrixElementList, Size, SparseMatrix,
    };
    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, SparseVector, VectorElementList,
    };

    #[test]
    fn test_binary_operator_reducer() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (1, 5, 1).into(),
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

        let mut product_vector =
            SparseVector::<u8>::new(&context, &matrix_size.row_height()).unwrap();

        let reducer = BinaryOperatorReducer::new();

        reducer
            .to_column_vector(
                &Plus::<u8>::new(),
                &matrix,
                &Assignment::new(),
                &mut product_vector,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);

        let mask_element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            // (5, 5).into(),
        ]);

        let mask = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &matrix_size.row_height(),
            &mask_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector =
            SparseVector::<u8>::new(&context, &matrix_size.row_height()).unwrap();

        reducer
            .to_colunm_vector_with_mask(
                &Plus::<u8>::new(),
                &matrix,
                &Assignment::new(),
                &mut product_vector,
                &mask,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_vector.get_element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&5).unwrap(), None);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);
    }
}
