use std::ffi::c_void;
use std::ptr;
use std::{mem::MaybeUninit, sync::Arc};

use suitesparse_graphblas_sys::{
    GrB_Index, GrB_Matrix, GrB_Matrix_deserialize, GxB_Matrix_deserialize,
};

use crate::context::CallGraphBlasContext;
use crate::index::{ElementIndex, IndexConversion};
use crate::{context::Context, error::SparseLinearAlgebraError};

pub unsafe fn deserialize_suitesparse_graphblas_sparse_matrix(
    context: &Arc<Context>,
    serialized_suitesparse_graphblas_sparse_matrix: &[u8],
) -> Result<GrB_Matrix, SparseLinearAlgebraError> {
    let mut suitesparse_graphblas_sparse_matrix: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
    let raw_pointer_to_serialized_suitesparse_graphblas_sparse_matrix: *const c_void =
        serialized_suitesparse_graphblas_sparse_matrix.as_ptr() as *const c_void;
    let size_of_serialized_suitesparse_graphblas_sparse_matrix: GrB_Index =
        serialized_suitesparse_graphblas_sparse_matrix
            .len()
            .to_graphblas_index()?;

    context.call_without_detailed_error_information(|| unsafe {
        GrB_Matrix_deserialize(
            suitesparse_graphblas_sparse_matrix.as_mut_ptr(),
            ptr::null_mut(),
            raw_pointer_to_serialized_suitesparse_graphblas_sparse_matrix,
            size_of_serialized_suitesparse_graphblas_sparse_matrix,
        )
    })?;
    // TODO: research if using GxB_Matrix_deserialize would enable retrieving detailed error information

    let suitesparse_graphblas_sparse_matrix =
        unsafe { suitesparse_graphblas_sparse_matrix.assume_init() };

    Ok(suitesparse_graphblas_sparse_matrix)
}

#[cfg(test)]
mod tests {
    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementList, GetSparseMatrixSize,
    };
    use crate::collections::sparse_matrix::{
        GetGraphblasSparseMatrix, MatrixElementList, SparseMatrix,
    };
    use crate::collections::{
        GraphblasCollectionSerializerUsingZstandardCompression,
        SerializeSuitesparseGraphblasSparseMatrix,
    };
    use crate::operators::binary_operator::First;

    use super::*;

    #[test]
    fn serialize_and_deserialize_suitesparse_graphblas_sparse_matrix() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (2, 5, 11).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            (10, 15).into(),
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        let zstd_serializer = GraphblasCollectionSerializerUsingZstandardCompression::new(
            context.clone(),
            crate::collections::ZstandardCompressionLevel::DEFAULT,
        )
        .unwrap();

        let serialized_matrix = unsafe {
            zstd_serializer
                .serialize_suitesparse_grapblas_sparse_matrix(matrix.graphblas_matrix())
                .unwrap()
        };

        let deserialized_graphblas_matrix = unsafe {
            deserialize_suitesparse_graphblas_sparse_matrix(&context, serialized_matrix).unwrap()
        };
        let deserialized_sparse_matrix = unsafe {
            SparseMatrix::<u8>::from_graphblas_matrix(context, deserialized_graphblas_matrix)
                .unwrap()
        };

        assert_eq!(
            matrix.element_list().unwrap(),
            deserialized_sparse_matrix.element_list().unwrap()
        )
    }
}
