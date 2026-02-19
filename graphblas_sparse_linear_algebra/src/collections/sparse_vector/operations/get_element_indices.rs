use crate::collections::collection::Collection;
use crate::collections::sparse_vector::sparse_vector::GetGraphblasSparseVector;
use crate::collections::sparse_vector::SparseVector;
use crate::context::CallGraphBlasContext;
use crate::error::GraphblasError;
use crate::error::GraphblasErrorType;
use crate::error::SparseLinearAlgebraError;
use crate::graphblas_bindings::{
    GrB_Vector_extractTuples_BOOL, GrB_Vector_extractTuples_FP32, GrB_Vector_extractTuples_FP64,
    GrB_Vector_extractTuples_INT16, GrB_Vector_extractTuples_INT32, GrB_Vector_extractTuples_INT64,
    GrB_Vector_extractTuples_INT8, GrB_Vector_extractTuples_UINT16,
    GrB_Vector_extractTuples_UINT32, GrB_Vector_extractTuples_UINT64,
    GrB_Vector_extractTuples_UINT8,
};
use crate::index::ElementIndex;
use crate::index::IndexConversion;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::{ConvertVector, ValueType};

pub trait GetSparseVectorElementIndices<T: ValueType> {
    fn element_indices(&self) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError>;
}

impl<T: ValueType + GetSparseVectorElementIndicesTyped<T>> GetSparseVectorElementIndices<T>
    for SparseVector<T>
{
    fn element_indices(&self) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError> {
        T::element_indices(self)
    }
}

pub trait GetSparseVectorElementIndicesTyped<T: ValueType> {
    fn element_indices(
        vector: &(impl GetGraphblasSparseVector + Collection),
    ) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError>;
}

// TODO: consider using an iterator - perhaps benchmark performance before switching
macro_rules! implement_get_element_indices {
    ($value_type:ty, $_graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetSparseVectorElementIndicesTyped<$value_type> for $value_type {
            fn element_indices(
                vector: &(impl GetGraphblasSparseVector + Collection),
            ) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError> {
                let number_of_stored_elements = vector.number_of_stored_elements()?;

                let mut indices = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                vector.context_ref().call(|| unsafe {
                    $get_element_function(
                        indices.as_mut_ptr(),
                        std::ptr::null_mut(),
                        &mut number_of_stored_and_returned_elements,
                        vector.graphblas_vector_ptr())
                }, unsafe{ &vector.graphblas_vector_ptr() })?;

                let length_of_element_list = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if length_of_element_list == number_of_stored_elements {
                        indices.set_len(length_of_element_list);
                    } else {
                        let err: SparseLinearAlgebraError = GraphblasError::new(GraphblasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values {}", number_of_stored_elements, length_of_element_list)).into();
                        return Err(err)
                    }
                };

                let indices = indices.to_type()?;
                Ok(indices)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_indices,
    GrB_Vector_extractTuples
);
