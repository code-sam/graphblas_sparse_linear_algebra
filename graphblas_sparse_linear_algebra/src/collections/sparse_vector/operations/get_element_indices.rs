use crate::collections::collection::Collection;
use crate::collections::sparse_vector::SparseVector;
use crate::{error::SparseLinearAlgebraError, index::ElementIndex, value_type::ValueType};

use super::VectorElementIndexIterator;

pub trait GetSparseVectorElementIndices<T: ValueType> {
    fn element_indices(&self) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError>;
}

impl<T: ValueType> GetSparseVectorElementIndices<T> for SparseVector<T> {
    fn element_indices(&self) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError> {
        let vector_element_index_iterator = VectorElementIndexIterator::new(&self).unwrap();
        let number_of_stored_elements = self.number_of_stored_elements()?;
        let mut element_indices: Vec<ElementIndex> = Vec::with_capacity(number_of_stored_elements);

        for element_index in vector_element_index_iterator.into_iter() {
            element_indices.push(element_index);
        }

        Ok(element_indices)
    }
}
