use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::{binary_operator::BinaryOperator, monoid::Monoid, semiring::Semiring};
use crate::value_type::ValueType;

use crate::graphblas_bindings::{
    GrB_Matrix_kronecker_BinaryOp, GrB_Matrix_kronecker_Monoid, GrB_Matrix_kronecker_Semiring,
};

use super::binary_operator::AccumulatorBinaryOperator;
use super::mask::MatrixMask;
use super::options::GetOptionsForOperatorWithMatrixArguments;

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for SemiringKroneckerProductOperator {}
unsafe impl Sync for SemiringKroneckerProductOperator {}

#[derive(Debug, Clone)]
pub struct SemiringKroneckerProductOperator {}

impl SemiringKroneckerProductOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait SemiringKroneckerProduct<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        multiplication_operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> SemiringKroneckerProduct<EvaluationDomain>
    for SemiringKroneckerProductOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        multiplication_operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_kronecker_Semiring(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    multiplication_operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_matrix_ref() },
        )?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MonoidKroneckerProductOperator {}

impl MonoidKroneckerProductOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait MonoidKroneckerProduct<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        multiplication_operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> MonoidKroneckerProduct<EvaluationDomain>
    for MonoidKroneckerProductOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        multiplication_operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_kronecker_Monoid(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    multiplication_operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_matrix_ref() },
        )?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BinaryOperatorKroneckerProductOperator {}

impl BinaryOperatorKroneckerProductOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait BinaryOperatorKroneckerProduct<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        multiplication_operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> BinaryOperatorKroneckerProduct<EvaluationDomain>
    for BinaryOperatorKroneckerProductOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        multiplication_operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_kronecker_BinaryOp(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    multiplication_operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.graphblas_descriptor(),
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

    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementList, GetSparseMatrixElementValue,
    };
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First, Times};

    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::operators::mask::SelectEntireMatrix;
    use crate::operators::options::OptionsForOperatorWithMatrixArguments;

    #[test]
    fn test_element_wisemultiplication() {
        let context = Context::init_default().unwrap();

        let operator = Times::<i32>::new();
        let options = OptionsForOperatorWithMatrixArguments::new_default();
        let element_wise_matrix_multiplier = BinaryOperatorKroneckerProductOperator::new();

        let height = 2;
        let width = 2;
        let size: Size = (height, width).into();

        let multiplier = SparseMatrix::<i32>::new(&context, &size).unwrap();
        let multiplicant = multiplier.to_owned();

        let mut product = SparseMatrix::<i32>::new(&context, &(4, 4).into()).unwrap();

        // Test multiplication of empty matrices
        element_wise_matrix_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireMatrix::new(&context),
                &options,
            )
            .unwrap();
        let element_list = product.element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.element_value(&1, &1).unwrap(), None); // NoValue

        let multiplier_element_list = MatrixElementList::<i32>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);
        let multiplier = SparseMatrix::<i32>::from_element_list(
            &context,
            &size,
            &multiplier_element_list,
            &First::<i32>::new(),
        )
        .unwrap();

        let multiplicant_element_list = MatrixElementList::<i32>::from_element_vector(vec![
            (0, 0, 5).into(),
            (1, 0, 6).into(),
            (0, 1, 7).into(),
            (1, 1, 8).into(),
        ]);
        let multiplicant = SparseMatrix::<i32>::from_element_list(
            &context,
            &size,
            &multiplicant_element_list,
            &First::<i32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        element_wise_matrix_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireMatrix::new(&context),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0, &0).unwrap(), 5);
        assert_eq!(product.element_value_or_default(&1, &0).unwrap(), 6);
        assert_eq!(product.element_value_or_default(&0, &1).unwrap(), 7);
        assert_eq!(product.element_value_or_default(&1, &1).unwrap(), 8);

        assert_eq!(product.element_value_or_default(&2, &0).unwrap(), 10);
        assert_eq!(product.element_value_or_default(&3, &0).unwrap(), 12);
        assert_eq!(product.element_value_or_default(&2, &1).unwrap(), 14);
        assert_eq!(product.element_value_or_default(&3, &1).unwrap(), 16);

        assert_eq!(product.element_value_or_default(&0, &2).unwrap(), 15);
        assert_eq!(product.element_value_or_default(&1, &2).unwrap(), 18);
        assert_eq!(product.element_value_or_default(&0, &3).unwrap(), 21);
        assert_eq!(product.element_value_or_default(&1, &3).unwrap(), 24);

        assert_eq!(product.element_value_or_default(&2, &2).unwrap(), 20);
        assert_eq!(product.element_value_or_default(&3, &2).unwrap(), 24);
        assert_eq!(product.element_value_or_default(&2, &3).unwrap(), 28);
        assert_eq!(product.element_value_or_default(&3, &3).unwrap(), 32);

        // // test the use of an accumulator
        // let accumulator = Plus::<i32, i32, i32>::new();
        // let matrix_multiplier_with_accumulator =
        //     ElementWiseMatrixMultiplicationBinaryOperator::<i32, i32, i32>::new(
        //         &operator,
        //         &options,
        //         Some(&accumulator),
        //     );

        // matrix_multiplier_with_accumulator
        //     .apply(&multiplier, &multiplicant, &mut product)
        //     .unwrap();

        // assert_eq!(product.get_element_value((0, 0).into()).unwrap(), 5 * 2);
        // assert_eq!(product.get_element_value((1, 0).into()).unwrap(), 12 * 2);
        // assert_eq!(product.get_element_value((0, 1).into()).unwrap(), 21 * 2);
        // assert_eq!(product.get_element_value((1, 1).into()).unwrap(), 32 * 2);

        // // test the use of a mask
        // let mask_element_list = MatrixElementList::<u8>::from_element_vector(vec![
        //     (0, 0, 3).into(),
        //     (1, 0, 0).into(),
        //     (1, 1, 1).into(),
        // ]);
        // let mask = SparseMatrix::<u8>::from_element_list(
        //     &context,
        //     &size,
        //     &mask_element_list,
        //     &First::<u8, u8, u8>::new(),
        // )
        // .unwrap();

        // let matrix_multiplier = ElementWiseMatrixMultiplicationBinaryOperator::<i32, i32, i32>::new(
        //     &operator, &options, None,
        // );

        // let mut product = SparseMatrix::<i32>::new(&context, &size).unwrap();

        // matrix_multiplier
        //     .apply_with_mask(&mask.into(), &multiplier, &multiplicant, &mut product)
        //     .unwrap();

        // assert_eq!(product.get_element_value((0, 0).into()).unwrap(), 5);
        // assert_eq!(product.get_element_value((1, 0).into()).unwrap(), 0);
        // assert_eq!(product.get_element_value((0, 1).into()).unwrap(), 0);
        // assert_eq!(product.get_element_value((1, 1).into()).unwrap(), 32);
    }
}
