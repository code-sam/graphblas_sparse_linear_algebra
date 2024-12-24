use suitesparse_graphblas_sys::GrB_Vector_removeElement;

use crate::{
    collections::sparse_vector::{GetGraphblasSparseVector, SparseVector},
    context::CallGraphBlasContext,
    error::SparseLinearAlgebraError,
    index::{ElementIndex, IndexConversion},
    value_type::ValueType,
};

pub trait DeleteSparseVectorElement {
    fn drop_element(
        &mut self,
        index_to_delete: ElementIndex,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> DeleteSparseVectorElement for SparseVector<T> {
    fn drop_element(
        &mut self,
        index_to_delete: ElementIndex,
    ) -> Result<(), SparseLinearAlgebraError> {
        drop_sparse_vector_element(self, index_to_delete)
    }
}

pub fn drop_sparse_vector_element(
    vector: &mut impl GetGraphblasSparseVector,
    index_to_delete: ElementIndex,
) -> Result<(), SparseLinearAlgebraError> {
    let index_to_delete = index_to_delete.to_graphblas_index()?;

    vector.context_ref().call(
        || unsafe { GrB_Vector_removeElement(vector.graphblas_vector(), index_to_delete) },
        unsafe { &vector.graphblas_vector() },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        collections::{
            sparse_vector::{
                operations::{DeleteSparseVectorElement, SetSparseVectorElement},
                SparseVector, VectorElement,
            },
            Collection,
        },
        context::Context,
        index::ElementCount,
    };

    #[test]
    fn remove_same_element_from_vector_twice() {
        let context = Context::init_default().unwrap();

        let length: ElementCount = 10;

        let mut sparse_vector = SparseVector::<i64>::new(context, length).unwrap();

        sparse_vector
            .set_element(VectorElement::from_pair(2, 3))
            .unwrap();
        sparse_vector
            .set_element(VectorElement::from_pair(4, 4))
            .unwrap();

        sparse_vector.drop_element(2).unwrap();
        sparse_vector.drop_element(2).unwrap();

        assert_eq!(sparse_vector.number_of_stored_elements().unwrap(), 1)
    }
}
