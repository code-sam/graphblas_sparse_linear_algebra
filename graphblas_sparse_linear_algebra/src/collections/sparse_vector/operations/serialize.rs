use core::slice;
use std::{ffi::c_void, mem::MaybeUninit};

use suitesparse_graphblas_sys::{GrB_Index, GrB_Vector, GxB_Vector_serialize};

use crate::context::CallGraphBlasContext;
use crate::index::IndexConversion;
use crate::{
    collections::GetGraphblasSerializerDescriptor, context::GetContext,
    error::SparseLinearAlgebraError, index::ElementIndex,
};

pub trait SerializeSuitesparseGraphblasSparseVector {
    unsafe fn serialize_suitesparse_grapblas_sparse_vector(
        &self,
        graphblas_sparse_vector: GrB_Vector,
    ) -> Result<&[u8], SparseLinearAlgebraError>;
}

pub unsafe fn serialize_suitesparse_grapblas_sparse_vector(
    serializer: &(impl GetGraphblasSerializerDescriptor + GetContext),
    suitesparse_graphblas_sparse_vector: GrB_Vector,
) -> Result<&[u8], SparseLinearAlgebraError> {
    let mut size_of_serialized_vector: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
    let mut serialized_vector_pointer: MaybeUninit<*mut c_void> = MaybeUninit::uninit();

    serializer.context_ref().call(
        || unsafe {
            GxB_Vector_serialize(
                serialized_vector_pointer.as_mut_ptr(),
                size_of_serialized_vector.as_mut_ptr(),
                suitesparse_graphblas_sparse_vector,
                serializer.graphblas_serializer_descriptor(),
            )
        },
        serializer.graphblas_serializer_descriptor_ref(),
    )?;

    let size_of_serialized_vector =
        ElementIndex::from_graphblas_index(unsafe { size_of_serialized_vector.assume_init() })?;
    let serialized_vector_pointer = unsafe { serialized_vector_pointer.assume_init() };

    let serialized_vector = unsafe {
        slice::from_raw_parts(
            serialized_vector_pointer as *const u8,
            size_of_serialized_vector,
        )
    };

    Ok(serialized_vector)
}
