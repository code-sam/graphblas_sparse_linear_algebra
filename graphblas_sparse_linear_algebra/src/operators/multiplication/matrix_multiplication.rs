use std::ptr;

use crate::collections::sparse_matrix::GraphblasSparseMatrixTrait;
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::{OperatorOptions, OperatorOptionsTrait};
use crate::operators::semiring::Semiring;
use crate::value_type::ValueType;

use crate::bindings_to_graphblas_implementation::GrB_mxm;

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for MatrixMultiplicationOperator {}
unsafe impl Sync for MatrixMultiplicationOperator {}

#[derive(Debug, Clone)]
pub struct MatrixMultiplicationOperator {}

impl MatrixMultiplicationOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait MultiplyMatrices<EvaluationDomain: ValueType> {
    // TODO: consider a version where the resulting product matrix is generated in the function body
    fn apply(
        &self,
        multiplier: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_mask(
        &self,
        multiplier: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> MultiplyMatrices<EvaluationDomain>
    for MatrixMultiplicationOperator
{
    // TODO: consider a version where the resulting product matrix is generated in the function body
    fn apply(
        &self,
        multiplier: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_mxm(
                    product.graphblas_matrix(),
                    ptr::null_mut(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_matrix_ref() },
        )?;

        Ok(())
    }

    fn apply_with_mask(
        &self,
        multiplier: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_mxm(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_matrix_ref() },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementList, GetMatrixElementValue, MatrixElementList,
        Size, SparseMatrix,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::binary_operator::{Plus, Times};
    use crate::operators::semiring::PlusTimes;

    #[test]
    fn test_multiplication_with_plus_times() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let semiring = PlusTimes::<f32>::new();
        let options = OperatorOptions::new_default();
        let matrix_multiplier = MatrixMultiplicationOperator::new();

        let height = 2;
        let width = 2;
        let size: Size = (height, width).into();

        let multiplier = SparseMatrix::<f32>::new(&context, &size).unwrap();
        let multiplicant = multiplier.clone();
        let mut product = multiplier.clone();

        // Test multiplication of empty matrices
        matrix_multiplier
            .apply(
                &multiplier,
                &semiring,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &OperatorOptions::new_default(),
            )
            .unwrap();
        let element_list = product.get_element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.get_element_value(&(1, 1).into()).unwrap(), None); // NoValue

        let multiplier_element_list = MatrixElementList::<f32>::from_element_vector(vec![
            (0, 0, 1.0).into(),
            (1, 0, 2.0).into(),
            (0, 1, 3.0).into(),
            (1, 1, 4.0).into(),
        ]);
        let multiplier = SparseMatrix::<f32>::from_element_list(
            &context,
            &size,
            &multiplier_element_list,
            &First::<f32>::new(),
        )
        .unwrap();

        let multiplicant_element_list = MatrixElementList::<f32>::from_element_vector(vec![
            (0, 0, 5.0).into(),
            (1, 0, 6.0).into(),
            (0, 1, 7.0).into(),
            (1, 1, 8.0).into(),
        ]);
        let multiplicant = SparseMatrix::<f32>::from_element_list(
            &context,
            &size,
            &multiplicant_element_list,
            &First::<f32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        matrix_multiplier
            .apply(
                &multiplier,
                &semiring,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        assert_eq!(
            product
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            23.
        );
        assert_eq!(
            product
                .get_element_value_or_default(&(1, 0).into())
                .unwrap(),
            34.
        );
        assert_eq!(
            product
                .get_element_value_or_default(&(0, 1).into())
                .unwrap(),
            31.
        );
        assert_eq!(
            product
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            46.
        );

        // TODO: this test is not generic over column/row storage format.
        // Equality checks should be done at a matrix level, since the ordering of the element list is not guaranteed.
        let expected_product = MatrixElementList::<f32>::from_element_vector(vec![
            (0, 0, 23.).into(),
            (0, 1, 31.).into(),
            (1, 0, 34.).into(),
            (1, 1, 46.).into(),
        ]);
        let product_element_list = product.get_element_list().unwrap();
        assert_eq!(expected_product, product_element_list);

        // test the use of an accumulator
        let accumulator = Plus::<f32>::new();
        let matrix_multiplier_with_accumulator = MatrixMultiplicationOperator::new();

        matrix_multiplier_with_accumulator
            .apply(
                &multiplier,
                &semiring,
                &multiplicant,
                &accumulator,
                &mut product,
                &options,
            )
            .unwrap();

        assert_eq!(
            product
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            23. * 2.
        );
        assert_eq!(
            product
                .get_element_value_or_default(&(1, 0).into())
                .unwrap(),
            34. * 2.
        );
        assert_eq!(
            product
                .get_element_value_or_default(&(0, 1).into())
                .unwrap(),
            31. * 2.
        );
        assert_eq!(
            product
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            46. * 2.
        );

        // test the use of a mask
        let mask_element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 3).into(),
            (1, 0, 0).into(),
            (1, 1, 1).into(),
        ]);
        let mask = SparseMatrix::<u8>::from_element_list(
            &context,
            &size,
            &mask_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let matrix_multiplier = MatrixMultiplicationOperator::new();

        let mut product = SparseMatrix::<f32>::new(&context, &size).unwrap();

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

        assert_eq!(
            product
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            23.
        );
        assert_eq!(product.get_element_value(&(1, 0).into()).unwrap(), None);
        assert_eq!(product.get_element_value(&(0, 1).into()).unwrap(), None);
        assert_eq!(
            product
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            46.
        );
    }
}
