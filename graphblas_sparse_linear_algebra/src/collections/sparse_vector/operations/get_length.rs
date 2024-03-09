use std::mem::MaybeUninit;

use suitesparse_graphblas_sys::{GrB_Index, GrB_Vector_size};

use crate::index::IndexConversion;
use crate::{
    collections::sparse_vector::{GetGraphblasSparseVector, SparseVector},
    context::{CallGraphBlasContext, GetContext},
    error::SparseLinearAlgebraError,
    index::ElementIndex,
    value_type::ValueType,
};

pub trait GetSparseVectorLength {
    fn length(&self) -> Result<ElementIndex, SparseLinearAlgebraError>;
}

impl<T: ValueType> GetSparseVectorLength for SparseVector<T> {
    fn length(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        sparse_vector_length(self)
    }
}

pub fn sparse_vector_length(
    vector: &impl GetGraphblasSparseVector,
) -> Result<ElementIndex, SparseLinearAlgebraError> {
    let mut length: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
    vector.context_ref().call(
        || unsafe { GrB_Vector_size(length.as_mut_ptr(), vector.graphblas_vector()) },
        unsafe { &vector.graphblas_vector() },
    )?;
    let length = unsafe { length.assume_init() };
    Ok(ElementIndex::from_graphblas_index(length)?)
}
