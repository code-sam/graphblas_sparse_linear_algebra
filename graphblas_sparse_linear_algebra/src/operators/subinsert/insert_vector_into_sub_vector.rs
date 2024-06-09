use crate::collections::sparse_vector::operations::GetSparseVectorLength;
use crate::collections::sparse_vector::{GetGraphblasSparseVector, SparseVector};
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::graphblas_bindings::GxB_Vector_subassign;
use crate::index::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::GetOperatorOptions;

use crate::value_type::ValueType;

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertVectorIntoSubVectorOperator {}
unsafe impl Sync for InsertVectorIntoSubVectorOperator {}

#[derive(Debug, Clone)]
pub struct InsertVectorIntoSubVectorOperator {}

impl InsertVectorIntoSubVectorOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait InsertVectorIntoSubVector<VectorToInsertInto>
where
    VectorToInsertInto: ValueType,
{
    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply(
        &self,
        vector_to_insert_into: &mut SparseVector<VectorToInsertInto>,
        indices_to_insert_into: &ElementIndexSelector,
        vector_to_insert: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<VectorToInsertInto>,
        mask_for_vector_to_insert_into: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<VectorToInsertInto: ValueType> InsertVectorIntoSubVector<VectorToInsertInto>
    for InsertVectorIntoSubVectorOperator
{
    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply(
        &self,
        vector_to_insert_into: &mut SparseVector<VectorToInsertInto>,
        indices_to_insert_into: &ElementIndexSelector,
        vector_to_insert: &impl GetGraphblasSparseVector,
        accumulator: &impl AccumulatorBinaryOperator<VectorToInsertInto>,
        mask_for_vector_to_insert_into: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = vector_to_insert_into.context();

        let number_of_indices_to_insert_into = indices_to_insert_into
            .number_of_selected_elements(vector_to_insert_into.length()?)?
            .to_graphblas_index()?;

        let indices_to_insert_into = indices_to_insert_into.to_graphblas_type()?;

        match indices_to_insert_into {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GxB_Vector_subassign(
                            GetGraphblasSparseVector::graphblas_vector(vector_to_insert_into),
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
                        GxB_Vector_subassign(
                            GetGraphblasSparseVector::graphblas_vector(vector_to_insert_into),
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
        FromVectorElementList, GetSparseVectorElementValue,
    };
    use crate::collections::sparse_vector::VectorElementList;
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::index::ElementIndex;
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::options::OperatorOptions;

    #[test]
    fn test_insert_vector_into_vector() {
        let context = Context::init_default().unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 10).into(),
            (5, 12).into(),
        ]);

        let vector_length: usize = 10;
        let mut vector = SparseVector::<u8>::from_element_list(
            context.clone(),
            vector_length,
            element_list.clone(),
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
            context.clone(),
            vector_to_insert_length,
            element_list_to_insert,
            &First::<u8>::new(),
        )
        .unwrap();

        let mask_element_list = VectorElementList::<bool>::from_element_vector(vec![
            // (1, 1, true).into(),
            (2, true).into(),
            (4, true).into(),
            // (5, true).into(),
        ]);
        let mask = SparseVector::<bool>::from_element_list(
            context.clone(),
            vector_to_insert_length,
            mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let indices_to_insert: Vec<ElementIndex> = (0..5).collect();
        let indices_to_insert = ElementIndexSelector::Index(&indices_to_insert);

        let insert_operator = InsertVectorIntoSubVectorOperator::new();

        insert_operator
            .apply(
                &mut vector,
                &indices_to_insert,
                &vector_to_insert,
                &Assignment::new(),
                &SelectEntireVector::new(context.clone()),
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
            context.clone(),
            vector_length,
            element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply(
                &mut vector,
                &indices_to_insert,
                &vector_to_insert,
                &Assignment::new(),
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
