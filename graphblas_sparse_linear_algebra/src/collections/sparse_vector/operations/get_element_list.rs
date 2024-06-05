use crate::collections::collection::Collection;
use crate::collections::sparse_vector::sparse_vector::GetGraphblasSparseVector;
use crate::collections::sparse_vector::SparseVector;
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::error::GraphblasError;
use crate::error::GraphblasErrorType;
use crate::graphblas_bindings::{
    GrB_Vector_extractTuples_BOOL, GrB_Vector_extractTuples_FP32, GrB_Vector_extractTuples_FP64,
    GrB_Vector_extractTuples_INT16, GrB_Vector_extractTuples_INT32, GrB_Vector_extractTuples_INT64,
    GrB_Vector_extractTuples_INT8, GrB_Vector_extractTuples_UINT16,
    GrB_Vector_extractTuples_UINT32, GrB_Vector_extractTuples_UINT64,
    GrB_Vector_extractTuples_UINT8,
};
use crate::index::ElementIndex;
use crate::index::IndexConversion;
use crate::value_type::ConvertVector;
use crate::{
    collections::sparse_vector::VectorElementList,
    error::SparseLinearAlgebraError,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};
use suitesparse_graphblas_sys::GrB_Index;

pub trait GetSparseVectorElementList<T: ValueType> {
    fn get_element_list(&self) -> Result<VectorElementList<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType + GetVectorElementListTyped<T>> GetSparseVectorElementList<T>
    for SparseVector<T>
{
    fn get_element_list(&self) -> Result<VectorElementList<T>, SparseLinearAlgebraError> {
        T::get_element_list(self)
    }
}

pub trait GetVectorElementListTyped<T: ValueType> {
    fn get_element_list(
        vector: &SparseVector<T>,
    ) -> Result<VectorElementList<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_list {
    ($value_type:ty, $graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetVectorElementListTyped<$value_type> for $value_type {
            fn get_element_list(
                vector: &SparseVector<$value_type>,
            ) -> Result<VectorElementList<$value_type>, SparseLinearAlgebraError> {
                let number_of_stored_elements = vector.number_of_stored_elements()?;

                let mut graphblas_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);
                let mut values: Vec<$graphblas_implementation_type> = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                vector.context_ref().call(|| unsafe {
                    $get_element_function(
                        graphblas_indices.as_mut_ptr(),
                        values.as_mut_ptr(),
                        &mut number_of_stored_and_returned_elements,
                        vector.graphblas_vector())
                }, unsafe{ &vector.graphblas_vector() })?;

                let length_of_element_list = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if length_of_element_list == number_of_stored_elements {
                        graphblas_indices.set_len(length_of_element_list);
                        values.set_len(length_of_element_list);
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

                let values = ConvertVector::<$graphblas_implementation_type, $value_type>::to_type(values)?;
                let element_list = VectorElementList::from_vectors(indices, values)?;
                Ok(element_list)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_list,
    GrB_Vector_extractTuples
);
