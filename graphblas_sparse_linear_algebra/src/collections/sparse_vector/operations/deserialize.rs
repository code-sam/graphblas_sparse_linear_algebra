use std::ffi::c_void;
use std::ptr;
use std::{mem::MaybeUninit, sync::Arc};

use suitesparse_graphblas_sys::{GrB_Index, GrB_Vector, GxB_Vector_deserialize};

use crate::collections::sparse_vector::SparseVector;
use crate::index::IndexConversion;
use crate::value_type::ValueType;
use crate::{context::Context, error::SparseLinearAlgebraError};

pub trait DeserializeSparseVector<T: ValueType> {
    unsafe fn deserialize_suitesparse_graphblas_sparse_vector(
        context: Arc<Context>,
        serialized_suitesparse_graphblas_sparse_matrix: &[u8],
    ) -> Result<SparseVector<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType> DeserializeSparseVector<T> for SparseVector<T> {
    unsafe fn deserialize_suitesparse_graphblas_sparse_vector(
        context: Arc<Context>,
        serialized_suitesparse_graphblas_sparse_matrix: &[u8],
    ) -> Result<SparseVector<T>, SparseLinearAlgebraError> {
        let graphblas_sparse_matrix = unsafe {
            deserialize_suitesparse_graphblas_sparse_vector(
                &context,
                serialized_suitesparse_graphblas_sparse_matrix,
            )
        }?;
        SparseVector::from_graphblas_vector(context.to_owned(), graphblas_sparse_matrix)
    }
}

pub unsafe fn deserialize_suitesparse_graphblas_sparse_vector(
    context: &Arc<Context>,
    serialized_suitesparse_graphblas_sparse_vector: &[u8],
) -> Result<GrB_Vector, SparseLinearAlgebraError> {
    let mut suitesparse_graphblas_sparse_vector: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
    let raw_pointer_to_serialized_suitesparse_graphblas_sparse_vector: *const c_void =
        serialized_suitesparse_graphblas_sparse_vector.as_ptr() as *const c_void;
    let size_of_serialized_suitesparse_graphblas_sparse_vector: GrB_Index =
        serialized_suitesparse_graphblas_sparse_vector
            .len()
            .to_graphblas_index()?;

    context.call_without_detailed_error_information(|| unsafe {
        GxB_Vector_deserialize(
            suitesparse_graphblas_sparse_vector.as_mut_ptr(),
            ptr::null_mut(),
            raw_pointer_to_serialized_suitesparse_graphblas_sparse_vector,
            size_of_serialized_suitesparse_graphblas_sparse_vector,
            ptr::null_mut(), // TODO: Could set the (max) number of threads
        )
    })?;
    // TODO: research if retrieving detailed error information is possible

    let suitesparse_graphblas_sparse_vector =
        unsafe { suitesparse_graphblas_sparse_vector.assume_init() };

    Ok(suitesparse_graphblas_sparse_vector)
}

#[cfg(test)]
mod tests {
    use crate::collections::sparse_vector::operations::{
        FromVectorElementList, GetSparseVectorElementList,
        SerializeSuitesparseGraphblasSparseVector,
    };
    use crate::collections::sparse_vector::{
        GetGraphblasSparseVector, SparseVector, VectorElementList,
    };
    use crate::collections::LZ4HighCompressionSerializer;
    use crate::operators::binary_operator::First;

    use super::*;

    #[test]
    fn serialize_and_deserialize_suitesparse_graphblas_sparse_vector() {
        let context = Context::init_default().unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (2, 10).into(),
            (2, 11).into(),
        ]);

        let vector = SparseVector::<u8>::from_element_list(
            context.clone(),
            15,
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        let zstd_serializer = LZ4HighCompressionSerializer::new(
            context.clone(),
            crate::collections::LZ4HighCompressionLevel::L2,
        )
        .unwrap();

        let serialized_vector = unsafe {
            zstd_serializer
                .serialize_suitesparse_grapblas_sparse_vector(vector.graphblas_vector_ptr())
                .unwrap()
        };

        let deserialized_sparse_vector = unsafe {
            SparseVector::<u8>::deserialize_suitesparse_graphblas_sparse_vector(
                context,
                serialized_vector,
            )
        }
        .unwrap();

        assert_eq!(
            vector.element_list().unwrap(),
            deserialized_sparse_vector.element_list().unwrap()
        )
    }
}
