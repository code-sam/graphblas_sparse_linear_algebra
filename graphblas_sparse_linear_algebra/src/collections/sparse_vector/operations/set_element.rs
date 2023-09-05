use crate::collections::sparse_vector::sparse_vector::GraphblasSparseVectorTrait;
use crate::collections::sparse_vector::SparseVector;
use crate::context::CallGraphBlasContext;
use crate::context::ContextTrait;
use crate::index::IndexConversion;
use crate::value_type::ConvertScalar;
use crate::{
    collections::sparse_vector::VectorElement,
    error::SparseLinearAlgebraError,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};

use crate::bindings_to_graphblas_implementation::{
    GrB_Vector_setElement_BOOL, GrB_Vector_setElement_FP32, GrB_Vector_setElement_FP64,
    GrB_Vector_setElement_INT16, GrB_Vector_setElement_INT32, GrB_Vector_setElement_INT64,
    GrB_Vector_setElement_INT8, GrB_Vector_setElement_UINT16, GrB_Vector_setElement_UINT32,
    GrB_Vector_setElement_UINT64, GrB_Vector_setElement_UINT8,
};

pub trait SetVectorElement<T: ValueType> {
    fn set_element(&mut self, element: VectorElement<T>) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType + SetGraphblasVectorElement<T>> SetVectorElement<T> for SparseVector<T> {
    fn set_element(&mut self, element: VectorElement<T>) -> Result<(), SparseLinearAlgebraError> {
        T::set_element(self, element)
    }
}

pub trait SetGraphblasVectorElement<T: ValueType> {
    fn set_element(
        vector: &mut SparseVector<T>,
        element: VectorElement<T>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_set_element_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ident, $add_element_function:ident) => {
        impl SetGraphblasVectorElement<$value_type> for $value_type {
            fn set_element(
                vector: &mut SparseVector<$value_type>,
                element: VectorElement<$value_type>,
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
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_set_element_for_built_in_type,
    GrB_Vector_setElement
);