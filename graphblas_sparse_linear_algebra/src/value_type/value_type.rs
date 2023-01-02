use crate::bindings_to_graphblas_implementation::{
    GrB_BOOL, GrB_FP32, GrB_FP64, GrB_INT16, GrB_INT32, GrB_INT64, GrB_INT8, GrB_Type, GrB_UINT16,
    GrB_UINT32, GrB_UINT64, GrB_UINT8,
};

use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types_and_graphblas_function;

use super::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_for_all_floating_point_value_types,
    implement_macro_for_all_graphblas_index_integers, implement_macro_for_all_integers,
};

pub trait ValueType {
    fn to_graphblas_type() -> GrB_Type;
}

macro_rules! implement_value_type_for_graphblas_built_in_type {
    ($value_type: ty, $graphblas_type_identifier: ident) => {
        impl ValueType for $value_type {
            fn to_graphblas_type() -> GrB_Type {
                unsafe { $graphblas_type_identifier }
            }
        }
    };
}

implement_macro_for_all_value_types_and_graphblas_function!(
    implement_value_type_for_graphblas_built_in_type,
    GrB
);

// pub trait FloatingPoint: ValueType {}

// macro_rules! implement_floating_point_for_value_type {
//     ($value_type: ty) => {
//         impl FloatingPoint for $value_type {}
//     };
// }

// implement_macro_for_all_floating_point_value_types!(implement_floating_point_for_value_type);

// pub trait Integer: ValueType {}

// macro_rules! implement_integer_for_value_type {
//     ($value_type: ty) => {
//         impl Integer for $value_type {}
//     };
// }

// implement_macro_for_all_integers!(implement_integer_for_value_type);

// pub trait GraphBlasIndexInteger: ValueType {}

// macro_rules! implement_graphblas_index_integer_for_value_type {
//     ($value_type: ty) => {
//         impl GraphBlasIndexInteger for $value_type {}
//     };
// }

// implement_macro_for_all_graphblas_index_integers!(implement_graphblas_index_integer_for_value_type);
