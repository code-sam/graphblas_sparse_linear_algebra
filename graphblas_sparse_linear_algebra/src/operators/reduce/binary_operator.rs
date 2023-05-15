use std::marker::PhantomData;
use std::ptr;

use crate::collections::sparse_matrix::GraphblasSparseMatrixTrait;
use crate::collections::sparse_vector::GraphblasSparseVectorTrait;
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::ValueType;

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_reduce_BinaryOp,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<EvaluationDomain: ValueType> Send for BinaryOperatorReducer<EvaluationDomain> {}
unsafe impl<EvaluationDomain: ValueType> Sync for BinaryOperatorReducer<EvaluationDomain> {}

#[derive(Debug, Clone)]
pub struct BinaryOperatorReducer<EvaluationDomain: ValueType> {
    _evaluation_domain: PhantomData<EvaluationDomain>,

    binary_operator: GrB_BinaryOp,
    accumulator: GrB_BinaryOp,
    options: GrB_Descriptor,
}

impl<EvaluationDomain: ValueType> BinaryOperatorReducer<EvaluationDomain> {
    pub fn new(
        binary_operator: &impl BinaryOperator<EvaluationDomain>,
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
    ) -> Self {
        Self {
            binary_operator: binary_operator.graphblas_type(),
            accumulator: accumulator.accumulator_graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _evaluation_domain: PhantomData,
        }
    }
}

pub trait ReduceWithBinaryOperator {
    fn to_vector(
        &self,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;

    fn to_vector_with_mask(
        &self,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ReduceWithBinaryOperator
    for BinaryOperatorReducer<EvaluationDomain>
{
    fn to_vector(
        &self,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_reduce_BinaryOp(
                    product.graphblas_vector(),
                    ptr::null_mut(),
                    self.accumulator,
                    self.binary_operator,
                    argument.graphblas_matrix(),
                    self.options,
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
    }

    fn to_vector_with_mask(
        &self,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_reduce_BinaryOp(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    self.accumulator,
                    self.binary_operator,
                    argument.graphblas_matrix(),
                    self.options,
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
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

        let reducer = BinaryOperatorReducer::new(
            &Plus::<u8>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        reducer.to_vector(&matrix, &mut product_vector).unwrap();

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
            .to_vector_with_mask(&matrix, &mut product_vector, &mask)
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
