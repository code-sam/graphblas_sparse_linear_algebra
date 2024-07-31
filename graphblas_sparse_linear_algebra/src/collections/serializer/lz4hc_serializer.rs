use std::{mem::MaybeUninit, sync::Arc};

use suitesparse_graphblas_sys::{
    GrB_Desc_Field_GxB_COMPRESSION, GrB_Descriptor, GrB_Descriptor_free, GrB_Descriptor_new,
    GrB_Matrix, GrB_Vector, GxB_COMPRESSION_LZ4HC, GxB_Desc_set,
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
pub enum LZ4HighCompressionLevel {
    DEFAULT,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
}

impl LZ4HighCompressionLevel {
    fn to_graphblas_descriptor_offset(&self) -> u32 {
        match self {
            LZ4HighCompressionLevel::DEFAULT => 9,
            LZ4HighCompressionLevel::L0 => 0,
            LZ4HighCompressionLevel::L1 => 1,
            LZ4HighCompressionLevel::L2 => 2,
            LZ4HighCompressionLevel::L3 => 3,
            LZ4HighCompressionLevel::L4 => 4,
            LZ4HighCompressionLevel::L5 => 5,
            LZ4HighCompressionLevel::L6 => 6,
            LZ4HighCompressionLevel::L7 => 7,
            LZ4HighCompressionLevel::L8 => 8,
            LZ4HighCompressionLevel::L9 => 9,
        }
    }
}

pub struct LZ4HighCompressionSerializer {
    context: Arc<Context>,
    compression_level: LZ4HighCompressionLevel,
    graphblas_descriptor: GrB_Descriptor,
}

impl GetContext for LZ4HighCompressionSerializer {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl GetGraphblasSerializerDescriptor for LZ4HighCompressionSerializer {
    unsafe fn graphblas_serializer_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }

    unsafe fn graphblas_serializer_descriptor_ref(&self) -> &GrB_Descriptor {
        &self.graphblas_descriptor
    }
}

impl LZ4HighCompressionSerializer {
    pub fn new(
        context: Arc<Context>,
        compression_level: LZ4HighCompressionLevel,
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
                    GrB_Desc_Field_GxB_COMPRESSION,
                    GxB_COMPRESSION_LZ4HC + compression_level.to_graphblas_descriptor_offset(),
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

impl Drop for LZ4HighCompressionSerializer {
    fn drop(&mut self) {
        let _ = self
            .context
            .call_without_detailed_error_information(|| unsafe {
                GrB_Descriptor_free(&mut self.graphblas_descriptor)
            });
    }
}

impl SerializeSuitesparseGraphblasSparseMatrix for LZ4HighCompressionSerializer {
    unsafe fn serialize_suitesparse_grapblas_sparse_matrix(
        &self,
        suitesparse_graphblas_sparse_matrix: GrB_Matrix,
    ) -> Result<&[u8], SparseLinearAlgebraError> {
        serialize_suitesparse_grapblas_sparse_matrix(self, suitesparse_graphblas_sparse_matrix)
    }
}

impl SerializeSuitesparseGraphblasSparseVector for LZ4HighCompressionSerializer {
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
