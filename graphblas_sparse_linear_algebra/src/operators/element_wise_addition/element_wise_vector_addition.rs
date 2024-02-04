use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::GetGraphblasDescriptor;
use crate::operators::{binary_operator::BinaryOperator, monoid::Monoid, semiring::Semiring};
use crate::value_type::ValueType;

use crate::graphblas_bindings::{
    GrB_Vector_eWiseAdd_BinaryOp, GrB_Vector_eWiseAdd_Monoid, GrB_Vector_eWiseAdd_Semiring,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Sync for ElementWiseVectorAdditionSemiringOperator {}
unsafe impl Send for ElementWiseVectorAdditionSemiringOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorAdditionSemiringOperator {}

impl ElementWiseVectorAdditionSemiringOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseVectorAdditionSemiringOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseVector + GetContext),
        operator: &impl Semiring<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyElementWiseVectorAdditionSemiringOperator<EvaluationDomain>
    for ElementWiseVectorAdditionSemiringOperator
{
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseVector + GetContext),
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
                GrB_Vector_eWiseAdd_Semiring(
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
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorAdditionMonoidOperator {}

impl ElementWiseVectorAdditionMonoidOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseVectorAdditionMonoidOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseVector + GetContext),
        operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyElementWiseVectorAdditionMonoidOperator<EvaluationDomain>
    for ElementWiseVectorAdditionMonoidOperator
{
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseVector + GetContext),
        operator: &impl Monoid<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Vector_eWiseAdd_Monoid(
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
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorAdditionBinaryOperator {}

impl ElementWiseVectorAdditionBinaryOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyElementWiseVectorAdditionBinaryOperator<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseVector + GetContext),
        operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyElementWiseVectorAdditionBinaryOperator<EvaluationDomain>
    for ElementWiseVectorAdditionBinaryOperator
{
    fn apply(
        &self,
        multiplier: &(impl GetGraphblasSparseVector + GetContext),
        operator: &impl BinaryOperator<EvaluationDomain>,
        multiplicant: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Vector_eWiseAdd_BinaryOp(
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
        FromVectorElementList, GetVectorElementList, GetVectorElementValue,
    };
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First, Plus, Times};
    use crate::operators::mask::SelectEntireVector;

    #[test]
    fn test_element_wise_addition() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let operator = Times::<i32>::new();
        let options = OperatorOptions::new_default();
        let element_wise_vector_multiplier = ElementWiseVectorAdditionBinaryOperator::new();

        let length = 4;

        let multiplier = SparseVector::<i32>::new(&context, &length).unwrap();
        let multiplicant = multiplier.to_owned();
        let mut product = multiplier.to_owned();

        // Test multiplication of empty matrices
        element_wise_vector_multiplier
            .apply(
                &multiplier,
                &operator,
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

        let multiplier_element_list = VectorElementList::<i32>::from_element_vector(vec![
            (0, 1).into(),
            (1, 2).into(),
            (2, 3).into(),
            (3, 4).into(),
        ]);
        let multiplier = SparseVector::<i32>::from_element_list(
            &context,
            &length,
            &multiplier_element_list,
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
            &context,
            &length,
            &multiplicant_element_list,
            &First::<i32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        element_wise_vector_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::<i32>::new(),
                &mut product,
                &SelectEntireVector::new(&context),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(&0).unwrap(), 5);
        assert_eq!(product.element_value_or_default(&1).unwrap(), 12);
        assert_eq!(product.element_value_or_default(&2).unwrap(), 21);
        assert_eq!(product.element_value_or_default(&3).unwrap(), 32);

        // test the use of an accumulator
        let accumulator = Plus::<i32>::new();
        let matrix_multiplier_with_accumulator = ElementWiseVectorAdditionBinaryOperator::new();

        matrix_multiplier_with_accumulator
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &accumulator,
                &mut product,
                &SelectEntireVector::new(&context),
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
            &context,
            &length,
            &mask_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let matrix_multiplier = ElementWiseVectorAdditionBinaryOperator::new();

        let mut product = SparseVector::<i32>::new(&context, &length).unwrap();

        matrix_multiplier
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::<i32>::new(),
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
