use std::ffi::c_void;
use std::ptr;
use std::{mem::MaybeUninit, sync::Arc};

use suitesparse_graphblas_sys::{GrB_Index, GrB_Matrix, GrB_Matrix_deserialize};

use crate::collections::sparse_matrix::SparseMatrix;
use crate::index::IndexConversion;
use crate::value_type::ValueType;
use crate::{context::Context, error::SparseLinearAlgebraError};

pub trait DeserializeSparseMatrix<T: ValueType> {
    unsafe fn deserialize_suitesparse_graphblas_sparse_matrix(
        context: Arc<Context>,
        serialized_suitesparse_graphblas_sparse_matrix: &[u8],
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType> DeserializeSparseMatrix<T> for SparseMatrix<T> {
    unsafe fn deserialize_suitesparse_graphblas_sparse_matrix(
        context: Arc<Context>,
        serialized_suitesparse_graphblas_sparse_matrix: &[u8],
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError> {
        let graphblas_sparse_matrix = unsafe {
            deserialize_suitesparse_graphblas_sparse_matrix(
                &context,
                serialized_suitesparse_graphblas_sparse_matrix,
            )
        }?;
        SparseMatrix::from_graphblas_matrix(context.to_owned(), graphblas_sparse_matrix)
    }
}

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
    use std::fs;

    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementList,
        SerializeSuitesparseGraphblasSparseMatrix, SetSparseMatrixElement,
    };
    use crate::collections::sparse_matrix::{
        GetGraphblasSparseMatrix, MatrixElementList, SparseMatrix,
    };
    use crate::collections::{LZ4HighCompressionSerializer, ZStandardSerializer};
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

        let zstd_serializer = LZ4HighCompressionSerializer::new(
            context.clone(),
            crate::collections::LZ4HighCompressionLevel::DEFAULT,
        )
        .unwrap();

        let serialized_matrix = unsafe {
            zstd_serializer
                .serialize_suitesparse_grapblas_sparse_matrix(matrix.graphblas_matrix_ptr())
                .unwrap()
        };

        let deserialized_sparse_matrix = unsafe {
            SparseMatrix::<u8>::deserialize_suitesparse_graphblas_sparse_matrix(
                context,
                &serialized_matrix,
            )
        }
        .unwrap();

        assert_eq!(
            matrix.element_list().unwrap(),
            deserialized_sparse_matrix.element_list().unwrap()
        )
    }

    // #[test]
    // fn serialize_and_deserialize_memory_usage() {
    //     let context = Context::init_default().unwrap();

    //     let n_element = 1000;
    //     let n_repeat = 1000;

    //     let mut matrix = SparseMatrix::<usize>::new(
    //         context.clone(),
    //         (n_element*10, n_element*2).into(),
    //     )
    //     .unwrap();

    //     for i in 0..n_element {
    //         matrix.set_value(i, i, i).unwrap();
    //         matrix.set_value(i+1, i, i+1).unwrap();
    //         matrix.set_value(i+2, i+1, i+2).unwrap();
    //     }

    //     let zstd_serializer = ZStandardSerializer::new(
    //         context.clone(),
    //         crate::collections::ZstandardCompressionLevel::DEFAULT,
    //     )
    //     .unwrap();

    //     for i in 0..n_repeat {
    //         let serialized_matrix = unsafe {
    //             zstd_serializer
    //                 .serialize_suitesparse_grapblas_sparse_matrix(matrix.graphblas_matrix_ptr())
    //                 .unwrap()
    //         };

    //         let deserialized_sparse_matrix = unsafe {
    //             SparseMatrix::<usize>::deserialize_suitesparse_graphblas_sparse_matrix(
    //                 context.clone(),
    //                 &serialized_matrix,
    //             )
    //         }
    //         .unwrap();

    //         assert_eq!(
    //             matrix.element_list().unwrap(),
    //             deserialized_sparse_matrix.element_list().unwrap()
    //         );

    //         // Read RSS from /proc/self/status (Linux only)
    //         if let Ok(status) = fs::read_to_string("/proc/self/status") {
    //             for line in status.lines() {
    //                 if line.starts_with("VmRSS:") {
    //                     println!("[{i}] {line}");
    //                     break;
    //                 }
    //             }
    //         }

    //         // // Read memory map breakdown from /proc/self/smaps_rollup (Linux only)
    //         // if let Ok(smaps) = fs::read_to_string("/proc/self/smaps_rollup") {
    //         //     let fields = [
    //         //         "Rss:",
    //         //         "Private_Dirty:",
    //         //         "Private_Clean:",
    //         //         "Shared_Dirty:",
    //         //         "Shared_Clean:",
    //         //         "Anonymous:",
    //         //     ];
    //         //     for line in smaps.lines() {
    //         //         if fields.iter().any(|f| line.starts_with(f)) {
    //         //             println!("  {line}");
    //         //         }
    //         //     }
    //         // }

    //     }

    //     // Read RSS from /proc/self/status (Linux only)
    //     if let Ok(status) = fs::read_to_string("/proc/self/status") {
    //         for line in status.lines() {
    //             if line.starts_with("VmRSS:") {
    //                 println!("[END] {line}");
    //                 break;
    //             }
    //         }
    //     }

    //     // // Read memory map breakdown from /proc/self/smaps_rollup (Linux only)
    //     // if let Ok(smaps) = fs::read_to_string("/proc/self/smaps_rollup") {
    //     //     let fields = [
    //     //         "Rss:",
    //     //         "Private_Dirty:",
    //     //         "Private_Clean:",
    //     //         "Shared_Dirty:",
    //     //         "Shared_Clean:",
    //     //         "Anonymous:",
    //     //     ];
    //     //     for line in smaps.lines() {
    //     //         if fields.iter().any(|f| line.starts_with(f)) {
    //     //             println!("  {line}");
    //     //         }
    //     //     }
    //     // }

    //     assert!(false)
    // }
}
