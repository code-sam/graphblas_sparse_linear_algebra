use std::{mem::MaybeUninit, sync::Arc};

use suitesparse_graphblas_sys::{
    GrB_Desc_Field_GxB_COMPRESSION, GrB_Descriptor, GrB_Descriptor_free, GrB_Descriptor_new,
    GrB_Matrix, GrB_Vector, GxB_COMPRESSION_ZSTD, GxB_Desc_set,
};

use crate::collections::sparse_matrix::operations::{
    serialize_suitesparse_grapblas_sparse_matrix, SerializeSuitesparseGraphblasSparseMatrix,
};
use crate::collections::sparse_vector::operations::{
    serialize_suitesparse_grapblas_sparse_vector, SerializeSuitesparseGraphblasSparseVector,
};
use crate::context::CallGraphBlasContext;
use crate::{
    context::{Context, GetContext},
    error::SparseLinearAlgebraError,
};

use super::GetGraphblasSerializerDescriptor;

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

pub struct ZStandardSerializer {
    context: Arc<Context>,
    compression_level: ZstandardCompressionLevel,
    graphblas_descriptor: GrB_Descriptor,
}

impl GetContext for ZStandardSerializer {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl GetGraphblasSerializerDescriptor for ZStandardSerializer {
    unsafe fn graphblas_serializer_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }

    unsafe fn graphblas_serializer_descriptor_ref(&self) -> &GrB_Descriptor {
        &self.graphblas_descriptor
    }
}

impl ZStandardSerializer {
    pub fn new(
        context: Arc<Context>,
        compression_level: ZstandardCompressionLevel,
    ) -> Result<Self, SparseLinearAlgebraError> {
        let mut graphblas_descriptor: MaybeUninit<GrB_Descriptor> = MaybeUninit::uninit();

        context.call_without_detailed_error_information(|| unsafe {
            GrB_Descriptor_new(graphblas_descriptor.as_mut_ptr())
        })?;

        let graphblas_descriptor = unsafe { graphblas_descriptor.assume_init() };

        context.call(
            || unsafe {
                GxB_Desc_set(
                    graphblas_descriptor,
                    GrB_Desc_Field_GxB_COMPRESSION as i32,
                    GxB_COMPRESSION_ZSTD + compression_level.to_graphblas_descriptor_offset(),
                )
            },
            &graphblas_descriptor,
        )?;

        Ok(Self {
            context,
            compression_level,
            graphblas_descriptor,
        })
    }
}

impl Drop for ZStandardSerializer {
    fn drop(&mut self) {
        let _ = self
            .context
            .call_without_detailed_error_information(|| unsafe {
                GrB_Descriptor_free(&mut self.graphblas_descriptor)
            });
    }
}

impl SerializeSuitesparseGraphblasSparseMatrix for ZStandardSerializer {
    unsafe fn serialize_suitesparse_grapblas_sparse_matrix(
        &self,
        suitesparse_graphblas_sparse_matrix: GrB_Matrix,
    ) -> Result<&[u8], SparseLinearAlgebraError> {
        serialize_suitesparse_grapblas_sparse_matrix(self, suitesparse_graphblas_sparse_matrix)
    }
}

impl SerializeSuitesparseGraphblasSparseVector for ZStandardSerializer {
    unsafe fn serialize_suitesparse_grapblas_sparse_vector(
        &self,
        suitesparse_graphblas_sparse_vector: GrB_Vector,
    ) -> Result<&[u8], SparseLinearAlgebraError> {
        serialize_suitesparse_grapblas_sparse_vector(self, suitesparse_graphblas_sparse_vector)
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::LZ4HighCompressionSerializer;

    use super::*;

    #[test]
    fn new_serializer() {
        let context = Context::init_default().unwrap();

        let _zstd_serializer = LZ4HighCompressionSerializer::new(
            context.clone(),
            crate::collections::LZ4HighCompressionLevel::DEFAULT,
        )
        .unwrap();

        assert!(true)
    }
}
