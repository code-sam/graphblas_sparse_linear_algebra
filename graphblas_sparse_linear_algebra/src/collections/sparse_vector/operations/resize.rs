use suitesparse_graphblas_sys::GrB_Vector_resize;

use crate::{
    collections::sparse_vector::{GetGraphblasSparseVector, SparseVector},
    context::{CallGraphBlasContext, GetContext},
    error::SparseLinearAlgebraError,
    index::{ElementIndex, IndexConversion},
    value_type::ValueType,
};

pub trait ResizeSparseVector {
    fn resize(&mut self, new_length: ElementIndex) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> ResizeSparseVector for SparseVector<T> {
    /// All elements of self with an index coordinate outside of the new size are dropped.
    fn resize(&mut self, new_length: ElementIndex) -> Result<(), SparseLinearAlgebraError> {
        resize_sparse_vector(self, new_length)
    }
}

/// All elements of self with an index coordinate outside of the new size are dropped.
pub fn resize_sparse_vector(
    vector: &mut impl GetGraphblasSparseVector,
    new_length: ElementIndex,
) -> Result<(), SparseLinearAlgebraError> {
    let new_length = new_length.to_graphblas_index()?;

    vector.context_ref().call(
        || unsafe { GrB_Vector_resize(vector.graphblas_vector(), new_length) },
        unsafe { &vector.graphblas_vector() },
    )?;
    Ok(())
}
