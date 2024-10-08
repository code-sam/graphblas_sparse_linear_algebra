use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::CallGraphBlasContext;
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::MatrixMask;
use crate::operators::options::GetOptionsForOperatorWithMatrixArguments;
use crate::operators::{binary_operator::BinaryOperator, monoid::Monoid, semiring::Semiring};
use crate::value_type::ValueType;

use crate::graphblas_bindings::{
    GrB_Matrix_eWiseAdd_BinaryOp, GrB_Matrix_eWiseAdd_Monoid, GrB_Matrix_eWiseAdd_Semiring,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Sync for ElementWiseMatrixAdditionSemiringOperator {}
unsafe impl Send for ElementWiseMatrixAdditionSemiringOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseMatrixAdditionSemiringOperator {}

impl ElementWiseMatrixAdditionSemiringOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseMatrixAdditionSemiring<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyElementWiseMatrixAdditionSemiring<EvaluationDomain>
    for ElementWiseMatrixAdditionSemiringOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context_ref();

        context.call(
            || unsafe {
                GrB_Matrix_eWiseAdd_Semiring(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;
        Ok(())
    }
}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Sync for ElementWiseMatrixAdditionMonoidOperator {}
unsafe impl Send for ElementWiseMatrixAdditionMonoidOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseMatrixAdditionMonoidOperator {}

impl ElementWiseMatrixAdditionMonoidOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseMatrixAdditionMonoidOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyElementWiseMatrixAdditionMonoidOperator<EvaluationDomain>
    for ElementWiseMatrixAdditionMonoidOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context_ref();

        context.call(
            || unsafe {
                GrB_Matrix_eWiseAdd_Monoid(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;

        Ok(())
    }
}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Sync for ElementWiseMatrixAdditionBinaryOperator {}
unsafe impl Send for ElementWiseMatrixAdditionBinaryOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseMatrixAdditionBinaryOperator {}

impl ElementWiseMatrixAdditionBinaryOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseMatrixAdditionBinaryOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyElementWiseMatrixAdditionBinaryOperator<EvaluationDomain>
    for ElementWiseMatrixAdditionBinaryOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseMatrix,
        operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context_ref();

        context.call(
            || unsafe {
                GrB_Matrix_eWiseAdd_BinaryOp(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
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
        FromMatrixElementList, GetSparseMatrixElementList, GetSparseMatrixElementValue,
    };
    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First, Plus, Times};
    use crate::operators::mask::SelectEntireMatrix;
    use crate::operators::options::OptionsForOperatorWithMatrixArguments;

    #[test]
    fn test_element_wise_multiplication() {
        let context = Context::init_default().unwrap();

        let operator = Times::<i32>::new();
        let options = OptionsForOperatorWithMatrixArguments::new_default();
        let element_wise_matrix_multiplier = ElementWiseMatrixAdditionBinaryOperator::new();

        let height = 2;
        let width = 2;
        let size: Size = (height, width).into();

        let multiplier = SparseMatrix::<i32>::new(context.clone(), size).unwrap();
        let multiplicant = multiplier.clone();
        let mut product = multiplier.clone();

        // Test multiplication of empty matrices
        element_wise_matrix_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::<i32>::new(),
                &mut product,
                &SelectEntireMatrix::new(context.clone()),
                &options,
            )
            .unwrap();
        let element_list = product.element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.element_value(1, 1).unwrap(), None); // NoValue

        let multiplier_element_list = MatrixElementList::<i32>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);
        let multiplier = SparseMatrix::<i32>::from_element_list(
            context.clone(),
            size,
            multiplier_element_list,
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
            context.clone(),
            size,
            multiplicant_element_list,
            &First::<i32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        element_wise_matrix_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::<i32>::new(),
                &mut product,
                &SelectEntireMatrix::new(context.clone()),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(0, 0).unwrap(), 5);
        assert_eq!(product.element_value_or_default(1, 0).unwrap(), 12);
        assert_eq!(product.element_value_or_default(0, 1).unwrap(), 21);
        assert_eq!(product.element_value_or_default(1, 1).unwrap(), 32);

        // test the use of an accumulator
        let accumulator = Plus::<i32>::new();
        let matrix_multiplier_with_accumulator = ElementWiseMatrixAdditionBinaryOperator::new();

        matrix_multiplier_with_accumulator
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &accumulator,
                &mut product,
                &SelectEntireMatrix::new(context.clone()),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(0, 0).unwrap(), 5 * 2);
        assert_eq!(product.element_value_or_default(1, 0).unwrap(), 12 * 2);
        assert_eq!(product.element_value_or_default(0, 1).unwrap(), 21 * 2);
        assert_eq!(product.element_value_or_default(1, 1).unwrap(), 32 * 2);

        // test the use of a mask
        let mask_element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 3).into(),
            (1, 0, 0).into(),
            (1, 1, 1).into(),
        ]);
        let mask = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            size,
            mask_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let matrix_multiplier = ElementWiseMatrixAdditionBinaryOperator::new();

        let mut product = SparseMatrix::<i32>::new(context, size).unwrap();

        matrix_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &accumulator,
                &mut product,
                &mask,
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(0, 0).unwrap(), 5);
        assert_eq!(product.element_value(1, 0).unwrap(), None);
        assert_eq!(product.element_value(0, 1).unwrap(), None);
        assert_eq!(product.element_value_or_default(1, 1).unwrap(), 32);
    }

    #[test]
    fn test_element_wise_addition() {
        let context = Context::init_default().unwrap();

        let operator = Plus::<i32>::new();
        let options = OptionsForOperatorWithMatrixArguments::new_default();
        let element_wise_matrix_adder = ElementWiseMatrixAdditionBinaryOperator::new();

        let height = 2;
        let width = 2;
        let size: Size = (height, width).into();

        let multiplier = SparseMatrix::<i32>::new(context.clone(), size).unwrap();
        let multiplicant = multiplier.clone();
        let mut product = multiplier.clone();

        // Test multiplication of empty matrices
        element_wise_matrix_adder
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::<i32>::new(),
                &mut product,
                &SelectEntireMatrix::new(context.clone()),
                &options,
            )
            .unwrap();
        let element_list = product.element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.element_value(1, 1).unwrap(), None); // NoValue

        let multiplier_element_list = MatrixElementList::<i32>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);
        let multiplier = SparseMatrix::<i32>::from_element_list(
            context.clone(),
            size,
            multiplier_element_list,
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
            context.clone(),
            size,
            multiplicant_element_list,
            &First::<i32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        element_wise_matrix_adder
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireMatrix::new(context.clone()),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(0, 0).unwrap(), 6);
        assert_eq!(product.element_value_or_default(1, 0).unwrap(), 8);
        assert_eq!(product.element_value_or_default(0, 1).unwrap(), 10);
        assert_eq!(product.element_value_or_default(1, 1).unwrap(), 12);
    }
}
