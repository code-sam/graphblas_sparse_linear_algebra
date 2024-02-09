use std::ptr;

use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::GetGraphblasDescriptor;

use crate::operators::semiring::Semiring;
use crate::value_type::ValueType;

use crate::graphblas_bindings::GrB_mxv;

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for MatrixVectorMultiplicationOperator {}
unsafe impl Sync for MatrixVectorMultiplicationOperator {}

#[derive(Debug, Clone)]
pub struct MatrixVectorMultiplicationOperator {}

impl MatrixVectorMultiplicationOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait MultiplyMatrixByVector<EvaluationDomain: ValueType> {
    // TODO: consider a version where the resulting product matrix is generated in the function body
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;

    // TODO: consider a version where the resulting product matrix is generated in the function body
    fn apply_with_mask(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> MultiplyMatrixByVector<EvaluationDomain>
    for MatrixVectorMultiplicationOperator
{
    // TODO: consider a version where the resulting product matrix is generated in the function body
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_mxv(
                    product.graphblas_vector(),
                    ptr::null_mut(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_vector(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
    }

    // TODO: consider a version where the resulting product matrix is generated in the function body
    fn apply_with_mask(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_mxv(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_vector(),
                    options.graphblas_descriptor(),
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

    use crate::collections::sparse_matrix::operations::FromMatrixElementList;
    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::collections::sparse_vector::operations::{
        FromVectorElementList, GetVectorElementList, GetVectorElementValue,
    };
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::Plus;
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::options::OperatorOptions;
    use crate::operators::semiring::PlusTimes;

    #[test]
    fn test_multiplication_with_plus_times() {
        let context = Context::init_default().unwrap();

        let semiring = PlusTimes::<f32>::new();
        let options = OperatorOptions::new_default();
        let matrix_multiplier = MatrixVectorMultiplicationOperator::new();

        let length = 2;
        let size: Size = (length, length).into();

        let multiplier = SparseMatrix::<f32>::new(&context, &size).unwrap();
        let multiplicant = SparseVector::<f32>::new(&context, &length).unwrap();
        let mut product = multiplicant.to_owned();

        // Test multiplication of empty matrices
        matrix_multiplier
            .apply_with_mask(
                &multiplier,
                &semiring,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireVector::new(&context),
                &options,
            )
            .unwrap();
        let element_list = product.get_element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.element_value(&1).unwrap(), None); // NoValue

        let multiplicant_element_list =
            VectorElementList::<f32>::from_element_vector(vec![(0, 1.0).into(), (1, 2.0).into()]);
        let multiplicant = SparseVector::<f32>::from_element_list(
            &context,
            &length,
            &multiplicant_element_list,
            &First::<f32>::new(),
        )
        .unwrap();

        let multiplier_element_list = MatrixElementList::<f32>::from_element_vector(vec![
            (0, 0, 5.0).into(),
            (1, 0, 6.0).into(),
            (0, 1, 7.0).into(),
            (1, 1, 8.0).into(),
        ]);
        let multiplier = SparseMatrix::<f32>::from_element_list(
            &context,
            &size,
            &multiplier_element_list,
            &First::<f32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        matrix_multiplier
            .apply_with_mask(
                &multiplier,
                &semiring,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireVector::new(&context),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0).unwrap(), 19.);
        assert_eq!(product.element_value_or_default(&1).unwrap(), 22.);

        // TODO: this test is not generic over column/row storage format.
        // Equality checks should be done at a matrix level, since the ordering of the element list is not guaranteed.
        let expected_product =
            VectorElementList::<f32>::from_element_vector(vec![(0, 19.).into(), (1, 22.).into()]);
        let product_element_list = product.get_element_list().unwrap();
        assert_eq!(expected_product, product_element_list);

        // test the use of an accumulator
        let accumulator = Plus::<f32>::new();
        let matrix_multiplier_with_accumulator = MatrixVectorMultiplicationOperator::new();

        matrix_multiplier_with_accumulator
            .apply_with_mask(
                &multiplier,
                &semiring,
                &multiplicant,
                &accumulator,
                &mut product,
                &SelectEntireVector::new(&context),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0).unwrap(), 19. * 2.);
        assert_eq!(product.element_value_or_default(&1).unwrap(), 22. * 2.);

        // test the use of a mask
        let mask_element_list =
            VectorElementList::<u8>::from_element_vector(vec![(0, 3).into(), (1, 0).into()]);
        let mask = SparseVector::<u8>::from_element_list(
            &context,
            &length,
            &mask_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let matrix_multiplier = MatrixVectorMultiplicationOperator::new();

        let mut product = SparseVector::<f32>::new(&context, &length).unwrap();

        matrix_multiplier
            .apply_with_mask(
                &multiplier,
                &semiring,
                &multiplicant,
                &accumulator,
                &mut product,
                &mask,
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0).unwrap(), 19.);
        assert_eq!(product.element_value(&1).unwrap(), None);
    }
}
