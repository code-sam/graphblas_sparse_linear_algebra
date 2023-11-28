use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::MatrixMask;
use crate::operators::options::OperatorOptionsTrait;
use crate::operators::{
    binary_operator::BinaryOperator, monoid::Monoid, options::OperatorOptions, semiring::Semiring,
};
use crate::value_type::ValueType;

use crate::graphblas_bindings::{
    GrB_Matrix_eWiseMult_BinaryOp, GrB_Matrix_eWiseMult_Monoid, GrB_Matrix_eWiseMult_Semiring,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Sync for ElementWiseMatrixMultiplicationSemiringOperator {}
unsafe impl Send for ElementWiseMatrixMultiplicationSemiringOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseMatrixMultiplicationSemiringOperator {}

impl ElementWiseMatrixMultiplicationSemiringOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseMatrixMultiplicationSemiring<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyElementWiseMatrixMultiplicationSemiring<EvaluationDomain>
    for ElementWiseMatrixMultiplicationSemiringOperator
{
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_eWiseMult_Semiring(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
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
unsafe impl Sync for ElementWiseMatrixMultiplicationMonoidOperator {}
unsafe impl Send for ElementWiseMatrixMultiplicationMonoidOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseMatrixMultiplicationMonoidOperator {}

impl ElementWiseMatrixMultiplicationMonoidOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseMatrixMultiplicationMonoidOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType>
    ApplyElementWiseMatrixMultiplicationMonoidOperator<EvaluationDomain>
    for ElementWiseMatrixMultiplicationMonoidOperator
{
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_eWiseMult_Monoid(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
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
unsafe impl Sync for ElementWiseMatrixMultiplicationBinaryOperator {}
unsafe impl Send for ElementWiseMatrixMultiplicationBinaryOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseMatrixMultiplicationBinaryOperator {}

impl ElementWiseMatrixMultiplicationBinaryOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseMatrixMultiplicationBinaryOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType>
    ApplyElementWiseMatrixMultiplicationBinaryOperator<EvaluationDomain>
    for ElementWiseMatrixMultiplicationBinaryOperator
{
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_eWiseMult_BinaryOp(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_matrix(),
                    multiplicant.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
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
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First, Plus, Times};
    use crate::operators::mask::SelectEntireMatrix;

    #[test]
    fn test_element_wisemultiplication() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let operator = Times::<i32>::new();
        let options = OperatorOptions::new_default();
        let element_wise_matrix_multiplier = ElementWiseMatrixMultiplicationBinaryOperator::new();

        let height = 2;
        let width = 2;
        let size: Size = (height, width).into();

        let multiplier = SparseMatrix::<i32>::new(&context, &size).unwrap();
        let multiplicant = multiplier.to_owned();
        let mut product = multiplier.to_owned();

        // Test multiplication of empty matrices
        element_wise_matrix_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::<i32>::new(),
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
                &Assignment::<i32>::new(),
                &mut product,
                &SelectEntireMatrix::new(&context),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0, &0).unwrap(), 5);
        assert_eq!(product.element_value_or_default(&1, &0).unwrap(), 12);
        assert_eq!(product.element_value_or_default(&0, &1).unwrap(), 21);
        assert_eq!(product.element_value_or_default(&1, &1).unwrap(), 32);

        // test the use of an accumulator
        let accumulator = Plus::<i32>::new();
        let matrix_multiplier_with_accumulator =
            ElementWiseMatrixMultiplicationBinaryOperator::new();

        matrix_multiplier_with_accumulator
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &accumulator,
                &mut product,
                &SelectEntireMatrix::new(&context),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0, &0).unwrap(), 5 * 2);
        assert_eq!(product.element_value_or_default(&1, &0).unwrap(), 12 * 2);
        assert_eq!(product.element_value_or_default(&0, &1).unwrap(), 21 * 2);
        assert_eq!(product.element_value_or_default(&1, &1).unwrap(), 32 * 2);

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

        let matrix_multiplier = ElementWiseMatrixMultiplicationBinaryOperator::new();

        let mut product = SparseMatrix::<i32>::new(&context, &size).unwrap();

        matrix_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &mask,
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0, &0).unwrap(), 5);
        assert_eq!(product.element_value_or_default(&1, &0).unwrap(), 0);
        assert_eq!(product.element_value_or_default(&0, &1).unwrap(), 0);
        assert_eq!(product.element_value_or_default(&1, &1).unwrap(), 32);
    }
}
