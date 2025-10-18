use std::{mem::MaybeUninit, sync::Arc};

use suitesparse_graphblas_sys::{
    GrB_Desc_Field_GxB_COMPRESSION, GrB_Descriptor, GrB_Descriptor_free, GrB_Descriptor_new,
    GrB_Matrix, GrB_Vector, GxB_COMPRESSION_LZ4, GxB_Desc_set,
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

pub struct LZ4Serializer {
    context: Arc<Context>,
    graphblas_descriptor: GrB_Descriptor,
}

impl GetContext for LZ4Serializer {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl GetGraphblasSerializerDescriptor for LZ4Serializer {
    unsafe fn graphblas_serializer_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }

    unsafe fn graphblas_serializer_descriptor_ref(&self) -> &GrB_Descriptor {
        &self.graphblas_descriptor
    }
}

impl LZ4Serializer {
    pub fn new(context: Arc<Context>) -> Result<Self, SparseLinearAlgebraError> {
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
                    GxB_COMPRESSION_LZ4,
                )
            },
            &graphblas_descriptor,
        )?;

        Ok(Self {
            context,
            graphblas_descriptor,
        })
    }
}

impl Drop for LZ4Serializer {
    fn drop(&mut self) {
        let _ = self
            .context
            .call_without_detailed_error_information(|| unsafe {
                GrB_Descriptor_free(&mut self.graphblas_descriptor)
            });
    }
}

impl SerializeSuitesparseGraphblasSparseMatrix for LZ4Serializer {
    unsafe fn serialize_suitesparse_grapblas_sparse_matrix(
        &self,
        suitesparse_graphblas_sparse_matrix: GrB_Matrix,
    ) -> Result<&[u8], SparseLinearAlgebraError> {
        serialize_suitesparse_grapblas_sparse_matrix(self, suitesparse_graphblas_sparse_matrix)
    }
}

impl SerializeSuitesparseGraphblasSparseVector for LZ4Serializer {
    unsafe fn serialize_suitesparse_grapblas_sparse_vector(
        &self,
        suitesparse_graphblas_sparse_vector: GrB_Vector,
    ) -> Result<&[u8], SparseLinearAlgebraError> {
        serialize_suitesparse_grapblas_sparse_vector(self, suitesparse_graphblas_sparse_vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_serializer() {
        let context = Context::init_default().unwrap();

        let _lz4_serializer = LZ4Serializer::new(context.clone()).unwrap();

        assert!(true)
    }
}
