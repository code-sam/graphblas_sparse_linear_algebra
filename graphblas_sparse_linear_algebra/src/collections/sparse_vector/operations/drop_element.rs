use suitesparse_graphblas_sys::GrB_Vector_removeElement;

use crate::{
    collections::sparse_vector::{GetGraphblasSparseVector, SparseVector},
    context::{CallGraphBlasContext, GetContext},
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

fn drop_sparse_vector_element(
    vector: &mut (impl GetGraphblasSparseVector + GetContext),
    index_to_delete: ElementIndex,
) -> Result<(), SparseLinearAlgebraError> {
    let index_to_delete = index_to_delete.to_graphblas_index()?;

    vector.context_ref().call(
        || unsafe { GrB_Vector_removeElement(vector.graphblas_vector(), index_to_delete) },
        unsafe { &vector.graphblas_vector() },
    )?;
    Ok(())
}
