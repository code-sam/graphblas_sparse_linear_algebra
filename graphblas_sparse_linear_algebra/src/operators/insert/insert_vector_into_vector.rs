use std::ptr;

use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::GetGraphblasDescriptor;



use crate::collections::sparse_vector::operations::sparse_vector_length;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::index::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::value_type::ValueType;

use crate::graphblas_bindings::GrB_Vector_assign;

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertVectorIntoVector {}
unsafe impl Sync for InsertVectorIntoVector {}

#[derive(Debug, Clone)]
pub struct InsertVectorIntoVector {}

impl InsertVectorIntoVector {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait InsertVectorIntoVectorTrait<AccumulatorEvaluationDomain>
where
    AccumulatorEvaluationDomain: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        vector_to_insert_into: &mut (impl GetGraphblasSparseVector + GetContext),
        indices_to_insert_into: &ElementIndexSelector,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask(
        &self,
        vector_to_insert_into: &mut (impl GetGraphblasSparseVector + GetContext),
        indices_to_insert_into: &ElementIndexSelector,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        mask_for_vector_to_insert_into: &(impl GetGraphblasSparseVector + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<AccumulatorEvaluationDomain: ValueType>
    InsertVectorIntoVectorTrait<AccumulatorEvaluationDomain> for InsertVectorIntoVector
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        vector_to_insert_into: &mut (impl GetGraphblasSparseVector + GetContext),
        indices_to_insert_into: &ElementIndexSelector,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = vector_to_insert_into.context();

        let number_of_indices_to_insert_into = indices_to_insert_into
            .number_of_selected_elements(sparse_vector_length(vector_to_insert_into)?)?
            .to_graphblas_index()?;

        let indices_to_insert_into = indices_to_insert_into.to_graphblas_type()?;

        match indices_to_insert_into {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GrB_Vector_assign(
                            vector_to_insert_into.graphblas_vector(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { vector_to_insert_into.graphblas_vector_ref() },
                )?;
            }

            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GrB_Vector_assign(
                            vector_to_insert_into.graphblas_vector(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            index,
                            number_of_indices_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { vector_to_insert_into.graphblas_vector_ref() },
                )?;
            }
        }

        Ok(())
    }

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask(
        &self,
        vector_to_insert_into: &mut (impl GetGraphblasSparseVector + GetContext),
        indices_to_insert_into: &ElementIndexSelector,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        mask_for_vector_to_insert_into: &(impl GetGraphblasSparseVector + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = vector_to_insert_into.context();

        let number_of_indices_to_insert_into = indices_to_insert_into
            .number_of_selected_elements(sparse_vector_length(vector_to_insert_into)?)?
            .to_graphblas_index()?;

        let indices_to_insert_into = indices_to_insert_into.to_graphblas_type()?;

        match indices_to_insert_into {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GrB_Vector_assign(
                            vector_to_insert_into.graphblas_vector(),
                            mask_for_vector_to_insert_into.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { vector_to_insert_into.graphblas_vector_ref() },
                )?;
            }

            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GrB_Vector_assign(
                            vector_to_insert_into.graphblas_vector(),
                            mask_for_vector_to_insert_into.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            index,
                            number_of_indices_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { vector_to_insert_into.graphblas_vector_ref() },
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_vector::operations::{
        FromVectorElementList, GetVectorElementValue,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};

    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::index::ElementIndex;

    #[test]
    fn test_insert_vector_into_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 10).into(),
            (5, 12).into(),
        ]);

        let vector_length: usize = 10;
        let mut vector = SparseVector::<u8>::from_element_list(
            &context,
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let element_list_to_insert = VectorElementList::<u8>::from_element_vector(vec![
            (1, 2).into(),
            (2, 3).into(),
            (4, 11).into(),
            // (5, 11).into(),
        ]);

        let vector_to_insert_length: usize = 5;
        let vector_to_insert = SparseVector::<u8>::from_element_list(
            &context,
            &vector_to_insert_length,
            &element_list_to_insert,
            &First::<u8>::new(),
        )
        .unwrap();

        let mask_element_list = VectorElementList::<bool>::from_element_vector(vec![
            // (1, 1, true).into(),
            (2, true).into(),
            (4, true).into(),
            (5, true).into(),
        ]);
        let mask = SparseVector::<bool>::from_element_list(
            &context,
            &vector_length,
            &mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let indices_to_insert: Vec<ElementIndex> = (0..5).collect();
        let indices_to_insert = ElementIndexSelector::Index(&indices_to_insert);

        let insert_operator = InsertVectorIntoVector::new();

        insert_operator
            .apply(
                &mut vector,
                &indices_to_insert,
                &vector_to_insert,
                &Assignment::<u8>::new(),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", vector);

        assert_eq!(vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(vector.element_value(&0).unwrap(), None);
        assert_eq!(vector.element_value_or_default(&2).unwrap(), 3);
        assert_eq!(vector.element_value_or_default(&4).unwrap(), 11);
        assert_eq!(vector.element_value_or_default(&5).unwrap(), 12);

        let mut vector = SparseVector::<u8>::from_element_list(
            &context,
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply_with_mask(
                &mut vector,
                &indices_to_insert,
                &vector_to_insert,
                &Assignment::<u8>::new(),
                &mask,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", vector);

        assert_eq!(vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(vector.element_value(&0).unwrap(), None);
        assert_eq!(vector.element_value_or_default(&2).unwrap(), 3);
        assert_eq!(vector.element_value_or_default(&4).unwrap(), 11);
        assert_eq!(vector.element_value_or_default(&5).unwrap(), 12);
        assert_eq!(vector.element_value_or_default(&1).unwrap(), 1);
    }
}
