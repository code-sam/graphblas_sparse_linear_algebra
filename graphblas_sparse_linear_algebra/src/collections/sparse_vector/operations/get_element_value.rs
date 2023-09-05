use suitesparse_graphblas_sys::{
    GrB_Vector_extractElement_BOOL, GrB_Vector_extractElement_FP32, GrB_Vector_extractElement_FP64,
    GrB_Vector_extractElement_INT16, GrB_Vector_extractElement_INT32,
    GrB_Vector_extractElement_INT64, GrB_Vector_extractElement_INT8,
    GrB_Vector_extractElement_UINT16, GrB_Vector_extractElement_UINT32,
    GrB_Vector_extractElement_UINT64, GrB_Vector_extractElement_UINT8,
};

use crate::collections::sparse_vector::sparse_vector::GraphblasSparseVectorTrait;
use crate::collections::sparse_vector::SparseVector;
use crate::context::CallGraphBlasContext;
use crate::context::ContextTrait;
use crate::error::GraphblasErrorType;
use crate::error::LogicErrorType;
use crate::error::SparseLinearAlgebraErrorType;
use crate::index::IndexConversion;
use crate::value_type::ConvertScalar;
use crate::{
    error::SparseLinearAlgebraError,
    index::ElementIndex,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};
use core::mem::MaybeUninit;

pub trait GetVectorElementValue<T: ValueType + Default> {
    fn get_element_value(
        &self,
        index: &ElementIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn get_element_value_or_default(
        &self,
        index: &ElementIndex,
    ) -> Result<T, SparseLinearAlgebraError>;
}

impl<T: ValueType + Default + GetGraphblasVectorElementValue<T>> GetVectorElementValue<T>
    for SparseVector<T>
{
    fn get_element_value(
        &self,
        index: &ElementIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError> {
        T::get_element_value(self, index)
    }

    fn get_element_value_or_default(
        &self,
        index: &ElementIndex,
    ) -> Result<T, SparseLinearAlgebraError> {
        T::get_element_value_or_default(self, index)
    }
}

pub trait GetGraphblasVectorElementValue<T: ValueType + Default> {
    fn get_element_value(
        vector: &SparseVector<T>,
        index: &ElementIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn get_element_value_or_default(
        vector: &SparseVector<T>,
        index: &ElementIndex,
    ) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_value_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetGraphblasVectorElementValue<$value_type> for $value_type {
            fn get_element_value(
                vector: &SparseVector<$value_type>,
                index: &ElementIndex,
            ) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                let mut value: MaybeUninit<$graphblas_implementation_type> = MaybeUninit::uninit();
                let index_to_get = index.to_graphblas_index()?;

                let result = vector.context_ref().call(
                    || unsafe {
                        $get_element_function(
                            value.as_mut_ptr(),
                            vector.graphblas_vector(),
                            index_to_get,
                        )
                    },
                    unsafe { &vector.graphblas_vector() },
                );

                match result {
                    Ok(_) => {
                        let value = unsafe { value.assume_init() };
                        Ok(Some(value.to_type()?))
                    }
                    Err(error) => match error.error_type() {
                        SparseLinearAlgebraErrorType::LogicErrorType(
                            LogicErrorType::GraphBlas(GraphblasErrorType::NoValue),
                        ) => Ok(None),
                        _ => Err(error),
                    },
                }
            }

            fn get_element_value_or_default(
                vector: &SparseVector<$value_type>,
                index: &ElementIndex,
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                Ok(<$value_type>::get_element_value(vector, index)?.unwrap_or_default())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_value_for_built_in_type,
    GrB_Vector_extractElement
);
