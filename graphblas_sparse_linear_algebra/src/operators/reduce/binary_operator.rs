use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::binary_operator::BinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::{
    GetGraphblasDescriptor, GetOptionsForMaskedOperatorWithMatrixArgument, WithTransposeMatrixArgument,
};
use crate::value_type::ValueType;

use crate::graphblas_bindings::GrB_Matrix_reduce_BinaryOp;

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
    fn to_colunm_vector(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetOptionsForMaskedOperatorWithMatrixArgument,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn to_row_vector(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &(impl GetOptionsForMaskedOperatorWithMatrixArgument + WithTransposeMatrixArgument),
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ReduceWithBinaryOperator<EvaluationDomain>
    for BinaryOperatorReducer
{
    fn to_colunm_vector(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetOptionsForMaskedOperatorWithMatrixArgument,
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
                    options.graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
    }

    fn to_row_vector(
        &self,
        operator: &impl BinaryOperator<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &(impl GetOptionsForMaskedOperatorWithMatrixArgument + WithTransposeMatrixArgument),
    ) -> Result<(), SparseLinearAlgebraError> {
        self.to_colunm_vector(
            operator,
            argument,
            accumulator,
            product,
            mask,
            &options.with_negated_transpose_matrix_argument(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::FromMatrixElementList;
    use crate::collections::sparse_vector::operations::{
        FromVectorElementList, GetVectorElementValue,
    };
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First, Plus};

    use crate::collections::sparse_matrix::{
        GetMatrixDimensions, MatrixElementList, Size, SparseMatrix,
    };
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::options::OptionsForMaskedOperatorWithMatrixArgument;

    #[test]
    fn test_binary_operator_reducer() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (1, 5, 1).into(),
            (2, 1, 2).into(),
            (4, 2, 4).into(),
            (5, 2, 5).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector =
            SparseVector::<u8>::new(&context, matrix_size.row_height_ref()).unwrap();

        let reducer = BinaryOperatorReducer::new();

        reducer
            .to_colunm_vector(
                &Plus::<u8>::new(),
                &matrix,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OptionsForMaskedOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.element_value(&9).unwrap(), None);

        let mask_element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            // (5, 5).into(),
        ]);

        let mask = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            matrix_size.row_height_ref(),
            &mask_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector =
            SparseVector::<u8>::new(&context, matrix_size.row_height_ref()).unwrap();

        reducer
            .to_colunm_vector(
                &Plus::<u8>::new(),
                &matrix,
                &Assignment::new(),
                &mut product_vector,
                &mask,
                &OptionsForMaskedOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_vector.element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.element_value(&5).unwrap(), None);
        assert_eq!(product_vector.element_value(&9).unwrap(), None);
    }
}
