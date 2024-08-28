use suitesparse_graphblas_sys::{
    GrB_Vector_extractElement_BOOL, GrB_Vector_extractElement_FP32, GrB_Vector_extractElement_FP64,
    GrB_Vector_extractElement_INT16, GrB_Vector_extractElement_INT32,
    GrB_Vector_extractElement_INT64, GrB_Vector_extractElement_INT8,
    GrB_Vector_extractElement_UINT16, GrB_Vector_extractElement_UINT32,
    GrB_Vector_extractElement_UINT64, GrB_Vector_extractElement_UINT8,
};

use crate::collections::sparse_vector::sparse_vector::GetGraphblasSparseVector;
use crate::collections::sparse_vector::SparseVector;
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::error::GraphblasErrorType;
use crate::error::LogicErrorType;
use crate::error::SparseLinearAlgebraErrorType;
use crate::index::IndexConversion;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types_and_graphblas_function;
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

pub trait GetSparseVectorElementValue<T: ValueType + Default> {
    fn element_value(&self, index: ElementIndex) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn element_value_or_default(&self, index: ElementIndex) -> Result<T, SparseLinearAlgebraError>;
}

impl<T: ValueType + Default + GetSparseVectorElementValueTyped<T>> GetSparseVectorElementValue<T>
    for SparseVector<T>
{
    fn element_value(&self, index: ElementIndex) -> Result<Option<T>, SparseLinearAlgebraError> {
        T::element_value(self, index)
    }

    fn element_value_or_default(&self, index: ElementIndex) -> Result<T, SparseLinearAlgebraError> {
        T::element_value_or_default(self, index)
    }
}

pub trait GetSparseVectorElementValueTyped<T: ValueType + Default> {
    fn element_value(
        vector: &SparseVector<T>,
        index: ElementIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn element_value_or_default(
        vector: &SparseVector<T>,
        index: ElementIndex,
    ) -> Result<T, SparseLinearAlgebraError>;
}


macro_rules! implement_get_element_value {
    ($value_type:ty) => {
        impl GetSparseVectorElementValueTyped<$value_type> for $value_type {
            fn element_value(
                vector: &SparseVector<$value_type>,
                index: ElementIndex
            ) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                unsafe { <$value_type as GetSparseVectorElementValueUntyped<$value_type>>::element_value(vector, index) }
            }

            fn element_value_or_default(
                vector: &SparseVector<$value_type>,
                index: ElementIndex,
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                unsafe { <$value_type as GetSparseVectorElementValueUntyped<$value_type>>::element_value_or_default(vector, index) }
            }
        }
    };
}

implement_macro_for_all_value_types!(
    implement_get_element_value
);

/// The value type T and the value type of the vector argument must match, otherwise the resulting element_value results from undefined behaviour.
pub trait GetSparseVectorElementValueUntyped<T: ValueType + Default> {
    unsafe fn element_value(
        vector: &(impl GetGraphblasSparseVector + GetContext),
        index: ElementIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    unsafe fn element_value_or_default(
        vector: &(impl GetGraphblasSparseVector + GetContext),
        index: ElementIndex
    ) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_value_unsafe {
    ($value_type:ty, $get_element_function:ident) => {
        impl GetSparseVectorElementValueUntyped<$value_type> for $value_type {
            unsafe fn element_value(
                vector: &(impl GetGraphblasSparseVector + GetContext),
                index: ElementIndex
            ) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                let mut value = MaybeUninit::uninit();
                let index_to_get = index.as_graphblas_index()?;

                let result = vector.context_ref().call(
                    || unsafe {
                        $get_element_function(
                            value.as_mut_ptr(),
                            vector.graphblas_vector(),
                            index_to_get
                        )
                    },
                    unsafe { &vector.graphblas_vector() },
                );

                match result {
                    Ok(_) => {
                        let value = unsafe { value.assume_init() };
                        // Casting to support isize and usize, redundant for other types. TODO: review performance improvements
                        Ok(Some(value.try_into().unwrap()))
                    }
                    Err(error) => match error.error_type() {
                        SparseLinearAlgebraErrorType::LogicErrorType(
                            LogicErrorType::GraphBlas(GraphblasErrorType::NoValue),
                        ) => Ok(None),
                        _ => Err(error),
                    },
                }
            }

            unsafe fn element_value_or_default(
                vector: &(impl GetGraphblasSparseVector + GetContext),
                index: ElementIndex
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                match <$value_type as GetSparseVectorElementValueUntyped<$value_type>>::element_value(vector, index)? {
                    Some(value) => Ok(value),
                    None => Ok(<$value_type>::default()),
                }
            }
        }
    };
}

implement_macro_for_all_value_types_and_graphblas_function!(
    implement_get_element_value_unsafe,
    GrB_Vector_extractElement
);

#[cfg(test)]
mod tests {
    // use super::*;

    // use crate::{collections::sparse_vector::{operations::SetSparseVectorElement, SparseVector}, context::Context, index::ElementCount};

    // #[test]
    // fn get_element_value_with_type_casting() {
    // let context = Context::init_default().unwrap();

    // let length: ElementCount = 10;
    // let index = 1;
    // let element_value = 1_000_000;

    // let mut sparse_vector = SparseVector::<i64>::new(context, length).unwrap();

    // sparse_vector.set_value(index, element_value).unwrap();

    // let value_as_u8 = u8::element_value_or_default(&sparse_vector, index).unwrap();

    // assert_eq!(value_as_u8, u8::MAX);
    // }
}
