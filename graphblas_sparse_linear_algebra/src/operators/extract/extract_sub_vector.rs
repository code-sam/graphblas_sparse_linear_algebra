use crate::collections::sparse_vector::operations::GetSparseVectorLength;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::index::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::{GetGraphblasDescriptor, GetOperatorOptions};
use crate::value_type::ValueType;

use crate::graphblas_bindings::GrB_Vector_extract;

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for SubVectorExtractor {}
unsafe impl Sync for SubVectorExtractor {}

#[derive(Debug, Clone)]
pub struct SubVectorExtractor {}

impl SubVectorExtractor {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ExtractSubVector<SubVector: ValueType> {
    /// Length of the mask must equal length of sub_vector
    fn apply(
        &self,
        vector_to_extract_from: &(impl GetGraphblasSparseVector + GetContext + GetSparseVectorLength),
        indices_to_extract: &ElementIndexSelector,
        accumulator: &impl AccumulatorBinaryOperator<SubVector>,
        sub_vector: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<SubVector: ValueType> ExtractSubVector<SubVector> for SubVectorExtractor {
    /// Length of the mask must equal length of sub_vector
    fn apply(
        &self,
        vector_to_extract_from: &(impl GetGraphblasSparseVector + GetContext + GetSparseVectorLength),
        indices_to_extract: &ElementIndexSelector,
        accumulator: &impl AccumulatorBinaryOperator<SubVector>,
        sub_vector: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = vector_to_extract_from.context();

        let number_of_indices_to_extract: ElementIndex;
        match indices_to_extract {
            ElementIndexSelector::Index(indices) => number_of_indices_to_extract = indices.len(),
            ElementIndexSelector::All => {
                number_of_indices_to_extract = vector_to_extract_from.length()?
            }
        }
        let number_of_indices_to_extract = number_of_indices_to_extract.to_graphblas_index()?;

        let indices_to_extract = indices_to_extract.to_graphblas_type()?;

        match indices_to_extract {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GrB_Vector_extract(
                            GetGraphblasSparseVector::graphblas_vector(sub_vector),
                            mask.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_extract_from.graphblas_vector(),
                            index.as_ptr(),
                            number_of_indices_to_extract,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { sub_vector.graphblas_vector_ref() },
                )?;
            }
            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GrB_Vector_extract(
                            GetGraphblasSparseVector::graphblas_vector(sub_vector),
                            mask.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_extract_from.graphblas_vector(),
                            index,
                            number_of_indices_to_extract,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { sub_vector.graphblas_vector_ref() },
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
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::options::OperatorOptions;

    #[test]
    fn test_vector_extraction() {
        let context = Context::init_default().unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            &10,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut sub_vector = SparseVector::<u8>::new(&context, &3).unwrap();

        let indices_to_extract: Vec<ElementIndex> = (0..3).collect();
        let indices_to_extract = ElementIndexSelector::Index(&indices_to_extract);

        let extractor = SubVectorExtractor::new();

        extractor
            .apply(
                &vector,
                &indices_to_extract,
                &Assignment::<u8>::new(),
                &mut sub_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        assert_eq!(sub_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(sub_vector.element_value_or_default(&2).unwrap(), 2);
    }

    #[test]
    fn test_vector_extraction_with_mask() {
        let context = Context::init_default().unwrap();

        let vector_element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (3, 3).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            &10,
            &vector_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mask_element_list = VectorElementList::<u8>::from_element_vector(vec![
            (0, 0).into(),
            (1, 1).into(),
            (2, 2).into(),
            (3, 3).into(),
            // (4, 4).into(),
            // (5, 5).into(),
            // (6, 6).into(),
        ]);

        let mask = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            &4,
            &mask_element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut sub_vector = SparseVector::<u8>::new(&context, &4).unwrap();

        let indices_to_extract: Vec<ElementIndex> = (0..4).collect();
        let indices_to_extract = ElementIndexSelector::Index(&indices_to_extract);
        // let indices_to_extract = ElementIndexSelector::All;

        let extractor = SubVectorExtractor::new();

        extractor
            .apply(
                &vector,
                &indices_to_extract,
                &Assignment::<u8>::new(),
                &mut sub_vector,
                &mask,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        assert_eq!(sub_vector.number_of_stored_elements().unwrap(), 3);
        assert_eq!(sub_vector.element_value_or_default(&1).unwrap(), 1);
        assert_eq!(sub_vector.element_value_or_default(&2).unwrap(), 2);
        assert_eq!(sub_vector.element_value_or_default(&3).unwrap(), 3);
    }
}
