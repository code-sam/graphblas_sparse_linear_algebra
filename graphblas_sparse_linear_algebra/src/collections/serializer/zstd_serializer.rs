use core::slice;
use std::ffi::c_char;
use std::{mem::MaybeUninit, os::raw::c_void, sync::Arc};

use suitesparse_graphblas_sys::{
    GrB_Desc_Field_GxB_COMPRESSION, GrB_Descriptor, GrB_Descriptor_free, GrB_Descriptor_new, GrB_Descriptor_set, GrB_Index, GrB_Matrix, GxB_COMPRESSION_ZSTD, GxB_Desc_set, GxB_Matrix_serialize
};

use crate::context::CallGraphBlasContext;
use crate::index::IndexConversion;
use crate::{
    context::{Context, GetContext},
    error::SparseLinearAlgebraError,
    index::ElementIndex,
};

use super::SerializeSuitesparseGraphblasSparseMatrix;

/// Higher levels target higher compression ratios but take increasingly more time.
pub enum ZstandardCompressionLevel {
    DEFAULT,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    L10,
    L11,
    L12,
    L13,
    L14,
    L15,
    L16,
    L17,
    L18,
    L19,
}

impl ZstandardCompressionLevel {
    fn to_graphblas_descriptor_offset(&self) -> u32 {
        match self {
            ZstandardCompressionLevel::DEFAULT => 1,
            ZstandardCompressionLevel::L1 => 1,
            ZstandardCompressionLevel::L2 => 2,
            ZstandardCompressionLevel::L3 => 3,
            ZstandardCompressionLevel::L4 => 4,
            ZstandardCompressionLevel::L5 => 5,
            ZstandardCompressionLevel::L6 => 6,
            ZstandardCompressionLevel::L7 => 7,
            ZstandardCompressionLevel::L8 => 8,
            ZstandardCompressionLevel::L9 => 9,
            ZstandardCompressionLevel::L10 => 10,
            ZstandardCompressionLevel::L11 => 11,
            ZstandardCompressionLevel::L12 => 12,
            ZstandardCompressionLevel::L13 => 13,
            ZstandardCompressionLevel::L14 => 14,
            ZstandardCompressionLevel::L15 => 15,
            ZstandardCompressionLevel::L16 => 16,
            ZstandardCompressionLevel::L17 => 17,
            ZstandardCompressionLevel::L18 => 18,
            ZstandardCompressionLevel::L19 => 19,
        }
    }
}

pub struct GraphblasCollectionSerializerUsingZstandardCompression {
    context: Arc<Context>,
    compression_level: ZstandardCompressionLevel,
    graphblas_descriptor: GrB_Descriptor,
}

impl GetContext for GraphblasCollectionSerializerUsingZstandardCompression {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl GraphblasCollectionSerializerUsingZstandardCompression {
    pub fn new(
        context: Arc<Context>,
        compression_level: ZstandardCompressionLevel,
    ) -> Result<Self, SparseLinearAlgebraError> {
        let mut graphblas_descriptor: MaybeUninit<GrB_Descriptor> = MaybeUninit::uninit();
        
        context.call_without_detailed_error_information(|| unsafe {
            GrB_Descriptor_new(graphblas_descriptor.as_mut_ptr())
        })?;

        let graphblas_descriptor = unsafe { graphblas_descriptor.assume_init() };
        
        context.call(|| unsafe {
            GxB_Desc_set(
                graphblas_descriptor,
                GrB_Desc_Field_GxB_COMPRESSION,
                GxB_COMPRESSION_ZSTD + compression_level.to_graphblas_descriptor_offset(),
            )
        }, &graphblas_descriptor)?;
        
        Ok(Self {
            context,
            compression_level,
            graphblas_descriptor,
        })
    }
}

impl Drop for GraphblasCollectionSerializerUsingZstandardCompression {
    fn drop(&mut self) {
        let _ = self
            .context
            .call_without_detailed_error_information(|| unsafe {
                GrB_Descriptor_free(&mut self.graphblas_descriptor)
            });
    }
}

impl SerializeSuitesparseGraphblasSparseMatrix
    for GraphblasCollectionSerializerUsingZstandardCompression
{
    unsafe fn serialize_suitesparse_grapblas_sparse_matrix(
        &self,
        suitesparse_graphblas_sparse_matrix: GrB_Matrix,
    ) -> Result<&[u8], SparseLinearAlgebraError> {
        let mut size_of_serialized_matrix: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        let mut serialized_matrix_pointer: MaybeUninit<*mut c_void> = MaybeUninit::uninit();

        self.context_ref().call(
            || unsafe {
                GxB_Matrix_serialize(
                    serialized_matrix_pointer.as_mut_ptr(),
                    size_of_serialized_matrix.as_mut_ptr(),
                    suitesparse_graphblas_sparse_matrix,
                    self.graphblas_descriptor,
                )
            },
            &self.graphblas_descriptor,
        )?;

        let size_of_serialized_matrix =
            ElementIndex::from_graphblas_index(unsafe { size_of_serialized_matrix.assume_init() })?;
        let serialized_matrix_pointer = unsafe { serialized_matrix_pointer.assume_init() };

        let serialized_matrix = unsafe {
            slice::from_raw_parts(
                serialized_matrix_pointer as *const u8,
                size_of_serialized_matrix,
            )
        };

        Ok(serialized_matrix)
    }
}
