use core::slice;
use std::{ffi::c_void, mem::MaybeUninit};

use suitesparse_graphblas_sys::{GrB_Index, GrB_Matrix, GxB_Matrix_serialize};

use crate::collections::GetGraphblasSerializerDescriptor;
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::index::IndexConversion;
use crate::{error::SparseLinearAlgebraError, index::ElementIndex};

pub trait SerializeSuitesparseGraphblasSparseMatrix {
    unsafe fn serialize_suitesparse_grapblas_sparse_matrix(
        &self,
        graphblas_sparse_matrix: GrB_Matrix,
    ) -> Result<Vec<u8>, SparseLinearAlgebraError>;
}

pub unsafe fn serialize_suitesparse_grapblas_sparse_matrix(
    serializer: &(impl GetGraphblasSerializerDescriptor + GetContext),
    suitesparse_graphblas_sparse_matrix: GrB_Matrix,
) -> Result<Vec<u8>, SparseLinearAlgebraError> {
    let mut size_of_serialized_matrix: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
    let mut serialized_matrix_pointer: MaybeUninit<*mut c_void> = MaybeUninit::uninit();

    serializer.context_ref().call(
        || unsafe {
            GxB_Matrix_serialize(
                serialized_matrix_pointer.as_mut_ptr(),
                size_of_serialized_matrix.as_mut_ptr(),
                suitesparse_graphblas_sparse_matrix,
                serializer.graphblas_serializer_descriptor(),
            )
        },
        &serializer.graphblas_serializer_descriptor(),
    )?;

    let size_of_serialized_matrix =
        ElementIndex::from_graphblas_index(unsafe { size_of_serialized_matrix.assume_init() })?;
    let serialized_matrix_pointer = unsafe { serialized_matrix_pointer.assume_init() };

    let serialized_matrix = unsafe {
        slice::from_raw_parts(
            serialized_matrix_pointer as *const u8,
            size_of_serialized_matrix,
        )
        .to_vec()
    };

    unsafe {
        (serializer.context_ref().memory_allocator_function_pointers.free)(serialized_matrix_pointer);
    }

    Ok(serialized_matrix)
}
