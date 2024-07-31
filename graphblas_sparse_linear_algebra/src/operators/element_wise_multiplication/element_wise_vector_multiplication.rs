use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::CallGraphBlasContext;
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::GetOperatorOptions;
use crate::operators::{binary_operator::BinaryOperator, monoid::Monoid, semiring::Semiring};
use crate::value_type::ValueType;

use crate::graphblas_bindings::{
    GrB_Vector_eWiseMult_BinaryOp, GrB_Vector_eWiseMult_Monoid, GrB_Vector_eWiseMult_Semiring,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Sync for ElementWiseVectorMultiplicationSemiringOperator {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiringOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorMultiplicationSemiringOperator {}

impl ElementWiseVectorMultiplicationSemiringOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseVectorMultiplicationSemiringOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseVector,
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType>
    ApplyElementWiseVectorMultiplicationSemiringOperator<EvaluationDomain>
    for ElementWiseVectorMultiplicationSemiringOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseVector,
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context_ref();

        context.call(
            || unsafe {
                GrB_Vector_eWiseMult_Semiring(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_vector(),
                    multiplicant.graphblas_vector(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }
}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorMultiplicationMonoidOperator {}

impl ElementWiseVectorMultiplicationMonoidOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseVectorMultiplicationMonoidOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseVector,
        operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType>
    ApplyElementWiseVectorMultiplicationMonoidOperator<EvaluationDomain>
    for ElementWiseVectorMultiplicationMonoidOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseVector,
        operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context_ref();

        context.call(
            || unsafe {
                GrB_Vector_eWiseMult_Monoid(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_vector(),
                    multiplicant.graphblas_vector(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }
}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorMultiplicationBinaryOperator {}

impl ElementWiseVectorMultiplicationBinaryOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseVectorMultiplicationBinaryOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseVector,
        operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType>
    ApplyElementWiseVectorMultiplicationBinaryOperator<EvaluationDomain>
    for ElementWiseVectorMultiplicationBinaryOperator
{
    fn apply(
        &self,
        multiplier: &impl GetGraphblasSparseVector,
        operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context_ref();

        context.call(
            || unsafe {
                GrB_Vector_eWiseMult_BinaryOp(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    multiplier.graphblas_vector(),
                    multiplicant.graphblas_vector(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_vector::operations::{
        FromVectorElementList, GetSparseVectorElementList, GetSparseVectorElementValue,
    };
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First, Plus, Times};
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::options::OperatorOptions;

    #[test]
    fn test_element_wisemultiplication() {
        let context = Context::init_default().unwrap();

        let operator = Times::<i32>::new();
        let options = OperatorOptions::new_default();
        let element_wise_vector_multiplier = ElementWiseVectorMultiplicationBinaryOperator::new();

        let length = 4;

        let multiplier = SparseVector::<i32>::new(context.clone(), length).unwrap();
        let multiplicant = multiplier.clone();
        let mut product = multiplier.clone();

        // Test multiplication of empty matrices
        element_wise_vector_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireVector::new(context.clone()),
                &options,
            )
            .unwrap();
        let element_list = product.element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.element_value(&1).unwrap(), None); // NoValue

        let multiplier_element_list = VectorElementList::<i32>::from_element_vector(vec![
            (0, 1).into(),
            (1, 2).into(),
            (2, 3).into(),
            (3, 4).into(),
        ]);
        let multiplier = SparseVector::<i32>::from_element_list(
            context.clone(),
            length,
            multiplier_element_list,
            &First::<i32>::new(),
        )
        .unwrap();

        let multiplicant_element_list = VectorElementList::<i32>::from_element_vector(vec![
            (0, 5).into(),
            (1, 6).into(),
            (2, 7).into(),
            (3, 8).into(),
        ]);
        let multiplicant = SparseVector::<i32>::from_element_list(
            context.clone(),
            length,
            multiplicant_element_list,
            &First::<i32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        element_wise_vector_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireVector::new(context.clone()),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0).unwrap(), 5);
        assert_eq!(product.element_value_or_default(&1).unwrap(), 12);
        assert_eq!(product.element_value_or_default(&2).unwrap(), 21);
        assert_eq!(product.element_value_or_default(&3).unwrap(), 32);

        // test the use of an accumulator
        let accumulator = Plus::<i32>::new();
        let matrix_multiplier_with_accumulator =
            ElementWiseVectorMultiplicationBinaryOperator::new();

        matrix_multiplier_with_accumulator
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &accumulator,
                &mut product,
                &SelectEntireVector::new(context.clone()),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0).unwrap(), 5 * 2);
        assert_eq!(product.element_value_or_default(&1).unwrap(), 12 * 2);
        assert_eq!(product.element_value_or_default(&2).unwrap(), 21 * 2);
        assert_eq!(product.element_value_or_default(&3).unwrap(), 32 * 2);

        // test the use of a mask
        let mask_element_list = VectorElementList::<u8>::from_element_vector(vec![
            (0, 3).into(),
            (1, 0).into(),
            (3, 1).into(),
        ]);
        let mask = SparseVector::<u8>::from_element_list(
            context.clone(),
            length,
            mask_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let matrix_multiplier = ElementWiseVectorMultiplicationBinaryOperator::new();

        let mut product = SparseVector::<i32>::new(context.clone(), length).unwrap();

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

        assert_eq!(product.element_value_or_default(&0).unwrap(), 5);
        assert_eq!(product.element_value(&1).unwrap(), None);
        assert_eq!(product.element_value(&2).unwrap(), None);
        assert_eq!(product.element_value_or_default(&3).unwrap(), 32);
    }
}
