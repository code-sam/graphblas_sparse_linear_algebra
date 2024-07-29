use std::{mem::MaybeUninit, sync::Arc};

use suitesparse_graphblas_sys::{
    GrB_Desc_Field_GxB_COMPRESSION, GrB_Desc_Value, GrB_Descriptor, GrB_Descriptor_free,
    GrB_Descriptor_set, GrB_Matrix, GxB_COMPRESSION_NONE, GxB_COMPRESSION_ZSTD,
};

use crate::{
    collections::sparse_matrix::GetGraphblasSparseMatrix, context::Context,
    error::SparseLinearAlgebraError,
};

// #define GxB_COMPRESSION_NONE -1     // no compression
// #define GxB_COMPRESSION_DEFAULT 0   // ZSTD (level 1)
// #define GxB_COMPRESSION_LZ4   1000  // LZ4
// #define GxB_COMPRESSION_LZ4HC 2000  // LZ4HC, with default level 9
// #define GxB_COMPRESSION_ZSTD  3000  // ZSTD, with default level 1

// pub trait SerializeGraphblasCollection<C> {
//     fn serialize_grapblas_sparse_matrix(
//         &self,
//         graphblas_sparse_matrix: &impl GetGraphblasSparseMatrix,
//     ) -> &[u8] {
//         todo!()
//     }
// }

pub trait SerializeSuitesparseGraphblasSparseMatrix {
    unsafe fn serialize_suitesparse_grapblas_sparse_matrix(
        &self,
        graphblas_sparse_matrix: GrB_Matrix,
    ) -> Result<&[u8], SparseLinearAlgebraError>;
}
