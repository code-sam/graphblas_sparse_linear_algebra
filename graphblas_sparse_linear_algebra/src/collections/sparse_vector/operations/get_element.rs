use crate::collections::sparse_vector::operations::get_element_value::GetVectorElementValue;
use crate::{
    collections::{
        sparse_matrix::operations::GetSparseMatrixElementValueTyped,
        sparse_vector::{SparseVector, VectorElement},
    },
    error::SparseLinearAlgebraError,
    index::ElementIndex,
    value_type::ValueType,
};

use super::GetVectorElementValueTyped;

pub trait GetVectorElement<T: ValueType> {
    fn get_element(
        &self,
        index: ElementIndex,
    ) -> Result<Option<VectorElement<T>>, SparseLinearAlgebraError>;
    fn get_element_or_default(
        &self,
        index: ElementIndex,
    ) -> Result<VectorElement<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType + Default + GetVectorElementValueTyped<T>> GetVectorElement<T>
    for SparseVector<T>
{
    fn get_element(
        &self,
        index: ElementIndex,
    ) -> Result<Option<VectorElement<T>>, SparseLinearAlgebraError> {
        match self.get_element_value(&index)? {
            Some(value) => Ok(Some(VectorElement::new(index, value))),
            None => Ok(None),
        }
    }

    fn get_element_or_default(
        &self,
        index: ElementIndex,
    ) -> Result<VectorElement<T>, SparseLinearAlgebraError> {
        Ok(VectorElement::new(
            index,
            self.get_element_value_or_default(&index)?,
        ))
    }
}
