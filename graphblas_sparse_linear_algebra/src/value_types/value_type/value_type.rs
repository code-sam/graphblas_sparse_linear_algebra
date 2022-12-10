use crate::bindings_to_graphblas_implementation::{
    GrB_BOOL, GrB_FP32, GrB_FP64, GrB_INT16, GrB_INT32, GrB_INT64, GrB_INT8, GrB_Type, GrB_UINT16,
    GrB_UINT32, GrB_UINT64, GrB_UINT8,
};
use crate::error::{LogicError, LogicErrorType, SparseLinearAlgebraError};
use crate::value_types::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_for_all_value_types, implement_macro_for_all_value_types_and_graphblas_function,
};

pub trait ValueType {}

macro_rules! implement_value_type_for_graphblas_built_in_type {
    ($value_type: ty) => {
        impl ValueType for $value_type {}
    };
}

implement_macro_for_all_value_types!(implement_value_type_for_graphblas_built_in_type);
