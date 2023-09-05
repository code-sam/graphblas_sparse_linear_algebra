use crate::bindings_to_graphblas_implementation::GrB_Index;
use crate::bindings_to_graphblas_implementation::{
    GrB_Vector_extractTuples_BOOL, GrB_Vector_extractTuples_FP32, GrB_Vector_extractTuples_FP64,
    GrB_Vector_extractTuples_INT16, GrB_Vector_extractTuples_INT32, GrB_Vector_extractTuples_INT64,
    GrB_Vector_extractTuples_INT8, GrB_Vector_extractTuples_UINT16,
    GrB_Vector_extractTuples_UINT32, GrB_Vector_extractTuples_UINT64,
    GrB_Vector_extractTuples_UINT8,
};
use crate::collections::collection::Collection;
use crate::collections::sparse_vector::GraphblasSparseVectorTrait;
use crate::collections::sparse_vector::SparseVector;
use crate::context::CallGraphBlasContext;
use crate::context::ContextTrait;
use crate::error::GraphblasError;
use crate::error::GraphblasErrorType;
use crate::index::IndexConversion;
use crate::{
    error::SparseLinearAlgebraError,
    index::ElementIndex,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};

pub trait GetVectorElementIndices<T: ValueType> {
    fn element_indices(&self) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError>;
}

impl<T: ValueType + GetGraphblasVectorElementIndices<T>> GetVectorElementIndices<T>
    for SparseVector<T>
{
    fn element_indices(&self) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError> {
        T::element_indices(self)
    }
}

pub trait GetGraphblasVectorElementIndices<T: ValueType> {
    fn element_indices(
        vector: &SparseVector<T>,
    ) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_indices {
    ($value_type:ty, $graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetGraphblasVectorElementIndices<$value_type> for $value_type {
            fn element_indices(
                vector: &SparseVector<$value_type>,
            ) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError> {
                let number_of_stored_elements = vector.number_of_stored_elements()?;

                let mut graphblas_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                vector.context_ref().call(|| unsafe {
                    $get_element_function(
                        graphblas_indices.as_mut_ptr(),
                        std::ptr::null_mut(),
                        &mut number_of_stored_and_returned_elements,
                        vector.graphblas_vector())
                }, unsafe{ &vector.graphblas_vector() })?;

                let length_of_element_list = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if length_of_element_list == number_of_stored_elements {
                        graphblas_indices.set_len(length_of_element_list);
                    } else {
                        let err: SparseLinearAlgebraError = GraphblasError::new(GraphblasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values {}",number_of_stored_elements, length_of_element_list)).into();
                        return Err(err)
                    }
                };

                let mut indices: Vec<ElementIndex> = Vec::with_capacity(length_of_element_list);

                for index in graphblas_indices.into_iter() {
                    indices.push(ElementIndex::from_graphblas_index(index)?);
                }
                Ok(indices)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_indices,
    GrB_Vector_extractTuples
);
