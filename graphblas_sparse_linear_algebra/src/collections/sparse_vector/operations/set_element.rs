use crate::collections::sparse_vector::sparse_vector::GetGraphblasSparseVector;
use crate::collections::sparse_vector::GetVectorElementIndex;
use crate::collections::sparse_vector::GetVectorElementValue;
use crate::collections::sparse_vector::SparseVector;
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::index::ElementIndex;
use crate::index::IndexConversion;
use crate::value_type::ConvertScalar;
use crate::{
    error::SparseLinearAlgebraError,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};

use crate::graphblas_bindings::{
    GrB_Vector_setElement_BOOL, GrB_Vector_setElement_FP32, GrB_Vector_setElement_FP64,
    GrB_Vector_setElement_INT16, GrB_Vector_setElement_INT32, GrB_Vector_setElement_INT64,
    GrB_Vector_setElement_INT8, GrB_Vector_setElement_UINT16, GrB_Vector_setElement_UINT32,
    GrB_Vector_setElement_UINT64, GrB_Vector_setElement_UINT8,
};

pub trait SetVectorElement<T: ValueType> {
    fn set_element(
        &mut self,
        element: &(impl GetVectorElementIndex + GetVectorElementValue<T>),
    ) -> Result<(), SparseLinearAlgebraError>;
    fn set_value(&mut self, index: &ElementIndex, value: T)
        -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType + SetVectorElementTyped<T>> SetVectorElement<T> for SparseVector<T> {
    fn set_element(
        &mut self,
        element: &(impl GetVectorElementIndex + GetVectorElementValue<T>),
    ) -> Result<(), SparseLinearAlgebraError> {
        T::set_element(self, element)
    }
    fn set_value(
        &mut self,
        index: &ElementIndex,
        value: T,
    ) -> Result<(), SparseLinearAlgebraError> {
        T::set_value(self, index, value)
    }
}

pub trait SetVectorElementTyped<T: ValueType> {
    fn set_element(
        vector: &mut (impl GetGraphblasSparseVector + GetContext),
        element: &(impl GetVectorElementIndex + GetVectorElementValue<T>),
    ) -> Result<(), SparseLinearAlgebraError>;
    fn set_value(
        vector: &mut (impl GetGraphblasSparseVector + GetContext),
        index: &ElementIndex,
        value: T,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_set_element_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ident, $add_element_function:ident) => {
        impl SetVectorElementTyped<$value_type> for $value_type {
            fn set_element(
                vector: &mut (impl GetGraphblasSparseVector + GetContext),
                element: &(impl GetVectorElementIndex + GetVectorElementValue<$value_type>),
            ) -> Result<(), SparseLinearAlgebraError> {
                let index_to_set = element.index().to_graphblas_index()?;
                let element_value = element.value().to_type()?;
                vector.context_ref().call(
                    || unsafe {
                        $add_element_function(
                            vector.graphblas_vector(),
                            element_value,
                            index_to_set,
                        )
                    },
                    unsafe { &vector.graphblas_vector() },
                )?;
                Ok(())
            }

            fn set_value(
                vector: &mut (impl GetGraphblasSparseVector + GetContext),
                index: &ElementIndex,
                value: $value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let index_to_set = index.as_graphblas_index()?;
                let element_value = value.to_type()?;
                vector.context_ref().call(
                    || unsafe {
                        $add_element_function(
                            vector.graphblas_vector(),
                            element_value,
                            index_to_set,
                        )
                    },
                    unsafe { &vector.graphblas_vector() },
                )?;
                Ok(())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_set_element_for_built_in_type,
    GrB_Vector_setElement
);
