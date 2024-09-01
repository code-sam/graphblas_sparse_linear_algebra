use std::{mem::MaybeUninit, sync::Arc};

use suitesparse_graphblas_sys::{
    GxB_Iterator, GxB_Iterator_get_BOOL, GxB_Iterator_get_FP32, GxB_Iterator_get_FP64,
    GxB_Iterator_get_INT16, GxB_Iterator_get_INT32, GxB_Iterator_get_INT64, GxB_Iterator_get_INT8,
    GxB_Iterator_get_UINT16, GxB_Iterator_get_UINT32, GxB_Iterator_get_UINT64,
    GxB_Iterator_get_UINT8, GxB_Iterator_new,
};

use crate::context::Context;
use crate::error::SparseLinearAlgebraError;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::ConvertScalar;
use crate::value_type::ValueType;

pub trait GetElementValueAtIteratorPosition<T: ValueType> {
    fn element_value_at_iterator_position(
        graphblas_iterator: GxB_Iterator,
    ) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_value_at_iterator_position {
    ($value_type:ty, $implementation_type:ty, $graphblas_operator:ident) => {
        impl GetElementValueAtIteratorPosition<$value_type> for $value_type {
            fn element_value_at_iterator_position(
                graphblas_iterator: GxB_Iterator,
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                let value: $implementation_type =
                    unsafe { $graphblas_operator(graphblas_iterator) };
                value.to_type()
            }
        }
    };
}
implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_value_at_iterator_position,
    GxB_Iterator_get
);

pub(crate) unsafe fn new_graphblas_iterator(
    context: &Arc<Context>,
) -> Result<GxB_Iterator, SparseLinearAlgebraError> {
    let mut iterator: MaybeUninit<GxB_Iterator> = MaybeUninit::uninit();

    context.call_without_detailed_error_information(|| unsafe {
        GxB_Iterator_new(iterator.as_mut_ptr())
    })?;

    let iterator = unsafe { iterator.assume_init() };
    return Ok(iterator);
}
