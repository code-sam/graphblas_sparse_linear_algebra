use rayon::prelude::*;

use crate::error::SparseLinearAlgebraError;
use crate::value_type::value_type::ValueType;

macro_rules! implement_macro_for_all_value_types {
    ($macro_identifier:ident) => {
        $macro_identifier!(bool);
        $macro_identifier!(i8);
        $macro_identifier!(i16);
        $macro_identifier!(i32);
        $macro_identifier!(i64);
        $macro_identifier!(u8);
        $macro_identifier!(u16);
        $macro_identifier!(u32);
        $macro_identifier!(u64);
        $macro_identifier!(f32);
        $macro_identifier!(f64);
        $macro_identifier!(isize);
        $macro_identifier!(usize);
    };
}
pub(crate) use implement_macro_for_all_value_types;

macro_rules! implement_macro_for_all_value_types_except_bool {
    ($macro_identifier:ident) => {
        // $macro_identifier!(bool);
        $macro_identifier!(i8);
        $macro_identifier!(i16);
        $macro_identifier!(i32);
        $macro_identifier!(i64);
        $macro_identifier!(u8);
        $macro_identifier!(u16);
        $macro_identifier!(u32);
        $macro_identifier!(u64);
        $macro_identifier!(f32);
        $macro_identifier!(f64);
        $macro_identifier!(isize);
        $macro_identifier!(usize);
    };
}
pub(crate) use implement_macro_for_all_value_types_except_bool;

macro_rules! implement_macro_for_all_value_types_except_bool_ {
    ($macro_identifier:ident) => {
        // $macro_identifier!(bool);
        $macro_identifier!(i8);
        $macro_identifier!(i16);
        $macro_identifier!(i32);
        $macro_identifier!(i64);
        $macro_identifier!(u8);
        $macro_identifier!(u16);
        $macro_identifier!(u32);
        $macro_identifier!(u64);
        $macro_identifier!(f32);
        $macro_identifier!(f64);
        // $macro_identifier!(isize);
        // $macro_identifier!(usize);
    };
}
pub(crate) use implement_macro_for_all_value_types_except_bool_;

macro_rules! implement_macro_for_all_integers {
    ($macro_identifier:ident) => {
        // $macro_identifier!(bool);
        $macro_identifier!(i8);
        $macro_identifier!(i16);
        $macro_identifier!(i32);
        $macro_identifier!(i64);
        $macro_identifier!(u8);
        $macro_identifier!(u16);
        $macro_identifier!(u32);
        $macro_identifier!(u64);
        // $macro_identifier!(f32);
        // $macro_identifier!(f64);
        $macro_identifier!(isize);
        $macro_identifier!(usize);
    };
}
pub(crate) use implement_macro_for_all_integers;

macro_rules! implement_macro_for_all_graphblas_index_integers {
    ($macro_identifier:ident) => {
        // $macro_identifier!(bool);
        // $macro_identifier!(i8);
        // $macro_identifier!(i16);
        $macro_identifier!(i32);
        $macro_identifier!(i64);
        // $macro_identifier!(u8);
        // $macro_identifier!(u16);
        // $macro_identifier!(u32);
        // $macro_identifier!(u64);
        // $macro_identifier!(f32);
        // $macro_identifier!(f64);
        // $macro_identifier!(isize);
        // $macro_identifier!(usize);
    };
}
pub(crate) use implement_macro_for_all_graphblas_index_integers;

macro_rules! implement_macro_for_all_floating_point_value_types {
    ($macro_identifier:ident) => {
        // $macro_identifier!(bool);
        // $macro_identifier!(i8);
        // $macro_identifier!(i16);
        // $macro_identifier!(i32);
        // $macro_identifier!(i64);
        // $macro_identifier!(u8);
        // $macro_identifier!(u16);
        // $macro_identifier!(u32);
        // $macro_identifier!(u64);
        $macro_identifier!(f32);
        $macro_identifier!(f64);
        // $macro_identifier!(isize);
        // $macro_identifier!(usize);
    };
}
pub(crate) use implement_macro_for_all_floating_point_value_types;

macro_rules! implement_macro_with_2_types_for_all_value_types {
    ($macro_identifier:ident) => {
        $macro_identifier!(bool, bool);
        $macro_identifier!(i8, i8);
        $macro_identifier!(i16, i16);
        $macro_identifier!(i32, i32);
        $macro_identifier!(i64, i64);
        $macro_identifier!(u8, u8);
        $macro_identifier!(u16, u16);
        $macro_identifier!(u32, u32);
        $macro_identifier!(u64, u64);
        $macro_identifier!(f32, f32);
        $macro_identifier!(f64, f64);
        $macro_identifier!(isize, isize);
        $macro_identifier!(usize, usize);
    };
}
pub(crate) use implement_macro_with_2_types_for_all_value_types;

macro_rules! implement_trait_for_all_value_types {
    ($trait:ty, $type_identifier:ident) => {
        unsafe impl $trait for $type_identifier<bool> {}
        unsafe impl $trait for $type_identifier<i8> {}
        unsafe impl $trait for $type_identifier<i16> {}
        unsafe impl $trait for $type_identifier<i32> {}
        unsafe impl $trait for $type_identifier<i64> {}
        unsafe impl $trait for $type_identifier<u8> {}
        unsafe impl $trait for $type_identifier<u16> {}
        unsafe impl $trait for $type_identifier<u32> {}
        unsafe impl $trait for $type_identifier<u64> {}
        unsafe impl $trait for $type_identifier<f32> {}
        unsafe impl $trait for $type_identifier<f64> {}
        unsafe impl $trait for $type_identifier<isize> {}
        unsafe impl $trait for $type_identifier<usize> {}
    };
}
pub(crate) use implement_trait_for_all_value_types;

macro_rules! implement_trait_for_2_type_data_type_and_all_value_types {
    ($trait:ty, $type_identifier:ident) => {
        unsafe impl $trait for $type_identifier<bool, bool> {}
        unsafe impl $trait for $type_identifier<i8, i8> {}
        unsafe impl $trait for $type_identifier<i16, i16> {}
        unsafe impl $trait for $type_identifier<i32, i32> {}
        unsafe impl $trait for $type_identifier<i64, i64> {}
        unsafe impl $trait for $type_identifier<u8, u8> {}
        unsafe impl $trait for $type_identifier<u16, u16> {}
        unsafe impl $trait for $type_identifier<u32, u32> {}
        unsafe impl $trait for $type_identifier<u64, u64> {}
        unsafe impl $trait for $type_identifier<f32, f32> {}
        unsafe impl $trait for $type_identifier<f64, f64> {}
        unsafe impl $trait for $type_identifier<isize, isize> {}
        unsafe impl $trait for $type_identifier<usize, usize> {}
    };
}
pub(crate) use implement_trait_for_2_type_data_type_and_all_value_types;

macro_rules! implement_trait_for_3_type_data_type_and_all_value_types {
    ($trait:ty, $type_identifier:ident) => {
        unsafe impl $trait for $type_identifier<bool, bool, bool> {}
        unsafe impl $trait for $type_identifier<i8, i8, i8> {}
        unsafe impl $trait for $type_identifier<i16, i16, i16> {}
        unsafe impl $trait for $type_identifier<i32, i32, i32> {}
        unsafe impl $trait for $type_identifier<i64, i64, i64> {}
        unsafe impl $trait for $type_identifier<u8, u8, u8> {}
        unsafe impl $trait for $type_identifier<u16, u16, u16> {}
        unsafe impl $trait for $type_identifier<u32, u32, u32> {}
        unsafe impl $trait for $type_identifier<u64, u64, u64> {}
        unsafe impl $trait for $type_identifier<f32, f32, f32> {}
        unsafe impl $trait for $type_identifier<f64, f64, f64> {}
        unsafe impl $trait for $type_identifier<isize, isize, isize> {}
        unsafe impl $trait for $type_identifier<usize, usize, usize> {}
    };
}
pub(crate) use implement_trait_for_3_type_data_type_and_all_value_types;

macro_rules! implement_trait_for_4_type_data_type_and_all_value_types {
    ($trait:ty, $type_identifier:ident) => {
        unsafe impl $trait for $type_identifier<bool, bool, bool, bool> {}
        unsafe impl $trait for $type_identifier<i8, i8, i8, i8> {}
        unsafe impl $trait for $type_identifier<i16, i16, i16, i16> {}
        unsafe impl $trait for $type_identifier<i32, i32, i32, i32> {}
        unsafe impl $trait for $type_identifier<i64, i64, i64, i64> {}
        unsafe impl $trait for $type_identifier<u8, u8, u8, u8> {}
        unsafe impl $trait for $type_identifier<u16, u16, u16, u16> {}
        unsafe impl $trait for $type_identifier<u32, u32, u32, u32> {}
        unsafe impl $trait for $type_identifier<u64, u64, u64, u64> {}
        unsafe impl $trait for $type_identifier<f32, f32, f32, f32> {}
        unsafe impl $trait for $type_identifier<f64, f64, f64, f64> {}
        unsafe impl $trait for $type_identifier<isize, isize, isize, isize> {}
        unsafe impl $trait for $type_identifier<usize, usize, usize, usize> {}
    };
}
pub(crate) use implement_trait_for_4_type_data_type_and_all_value_types;

macro_rules! implement_2_type_macro_for_all_value_types_and_untyped_graphblas_function {
    ($macro_identifier:ident, $graphblas_identifier:ident) => {
        $macro_identifier!(bool, bool, $graphblas_identifier);
        $macro_identifier!(i8, i8, $graphblas_identifier);
        $macro_identifier!(i16, i16, $graphblas_identifier);
        $macro_identifier!(i32, i32, $graphblas_identifier);
        $macro_identifier!(i64, i64, $graphblas_identifier);
        $macro_identifier!(u8, u8, $graphblas_identifier);
        $macro_identifier!(u16, u16, $graphblas_identifier);
        $macro_identifier!(u32, u32, $graphblas_identifier);
        $macro_identifier!(u64, u64, $graphblas_identifier);
        $macro_identifier!(f32, f32, $graphblas_identifier);
        $macro_identifier!(f64, f64, $graphblas_identifier);
        $macro_identifier!(isize, isize, $graphblas_identifier);
        $macro_identifier!(usize, usize, $graphblas_identifier);
    };
}
pub(crate) use implement_2_type_macro_for_all_value_types_and_untyped_graphblas_function;

macro_rules! implement_macro_with_custom_input_version_1_for_all_value_types {
    ($macro_identifier:ident, $trait:ident, $function_identifier_1:ident, $function_identifier_2:ident, $graphblas_identifier:ident) => {
        $macro_identifier!(
            bool,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            i8,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            i16,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            i32,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            i64,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            u8,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            u16,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            u32,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            u64,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            f32,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            f64,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            isize,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
        $macro_identifier!(
            usize,
            $trait,
            $function_identifier_1,
            $function_identifier_2,
            $graphblas_identifier
        );
    };
}
pub(crate) use implement_macro_with_custom_input_version_1_for_all_value_types;

macro_rules! implement_1_type_macro_for_all_value_types_and_typed_graphblas_function {
    ($macro_identifier:ident, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!(bool, [<$graphblas_identifier _BOOL>]);
            $macro_identifier!(i8, [<$graphblas_identifier _INT8>]);
            $macro_identifier!(i16, [<$graphblas_identifier _INT16>]);
            $macro_identifier!(i32, [<$graphblas_identifier _INT32>]);
            $macro_identifier!(i64, [<$graphblas_identifier _INT64>]);
            $macro_identifier!(u8, [<$graphblas_identifier _UINT8>]);
            $macro_identifier!(u16, [<$graphblas_identifier _UINT16>]);
            $macro_identifier!(u32, [<$graphblas_identifier _UINT32>]);
            $macro_identifier!(u64, [<$graphblas_identifier _UINT64>]);
            $macro_identifier!(f32, [<$graphblas_identifier _FP32>]);
            $macro_identifier!(f64, [<$graphblas_identifier _FP64>]);
            // $macro_identifier!(isize, id_isize!($graphblas_identifier));
            // $macro_identifier!(usize, id_usize!($graphblas_identifier));
            graphblas_sparse_linear_algebra_proc_macros::implement_1_type_macro_for_isize_and_typed_graphblas_function!($macro_identifier, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_1_type_macro_for_usize_and_typed_graphblas_function!($macro_identifier, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_1_type_macro_for_all_value_types_and_typed_graphblas_function;

macro_rules! implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type {
    ($macro_identifier:ident, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, [<$graphblas_identifier _BOOL>]);
            $macro_identifier!(i8, i8, [<$graphblas_identifier _INT8>]);
            $macro_identifier!(i16, i16, [<$graphblas_identifier _INT16>]);
            $macro_identifier!(i32, i32, [<$graphblas_identifier _INT32>]);
            $macro_identifier!(i64, i64, [<$graphblas_identifier _INT64>]);
            $macro_identifier!(u8, u8, [<$graphblas_identifier _UINT8>]);
            $macro_identifier!(u16, u16, [<$graphblas_identifier _UINT16>]);
            $macro_identifier!(u32, u32, [<$graphblas_identifier _UINT32>]);
            $macro_identifier!(u64, u64, [<$graphblas_identifier _UINT64>]);
            $macro_identifier!(f32, f32, [<$graphblas_identifier _FP32>]);
            $macro_identifier!(f64, f64, [<$graphblas_identifier _FP64>]);
            // $macro_identifier!(isize, isize, id_isize!($graphblas_identifier));
            // $macro_identifier!(usize, usize, id_usize!($graphblas_identifier));
            graphblas_sparse_linear_algebra_proc_macros::implement_1_type_macro_for_isize_and_typed_graphblas_function_with_implementation_type!($macro_identifier, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_1_type_macro_for_usize_and_typed_graphblas_function_with_implementation_type!($macro_identifier, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;

macro_rules! implement_1_type_macro_for_all_value_types_and_2_typed_graphblas_functions_with_implementation_type {
    ($macro_identifier:ident, $graphblas_identifier_1:ident, $graphblas_identifier_2:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, [<$graphblas_identifier_1 _BOOL>], [<$graphblas_identifier_2 _BOOL>]);
            $macro_identifier!(i8, i8, [<$graphblas_identifier_1 _INT8>], [<$graphblas_identifier_2 _INT8>]);
            $macro_identifier!(i16, i16, [<$graphblas_identifier_1 _INT16>], [<$graphblas_identifier_2 _INT16>]);
            $macro_identifier!(i32, i32, [<$graphblas_identifier_1 _INT32>], [<$graphblas_identifier_2 _INT32>]);
            $macro_identifier!(i64, i64, [<$graphblas_identifier_1 _INT64>], [<$graphblas_identifier_2 _INT64>]);
            $macro_identifier!(u8, u8, [<$graphblas_identifier_1 _UINT8>], [<$graphblas_identifier_2 _UINT8>]);
            $macro_identifier!(u16, u16, [<$graphblas_identifier_1 _UINT16>], [<$graphblas_identifier_2 _UINT16>]);
            $macro_identifier!(u32, u32, [<$graphblas_identifier_1 _UINT32>], [<$graphblas_identifier_2 _UINT32>]);
            $macro_identifier!(u64, u64, [<$graphblas_identifier_1 _UINT64>], [<$graphblas_identifier_2 _UINT64>]);
            $macro_identifier!(f32, f32, [<$graphblas_identifier_1 _FP32>], [<$graphblas_identifier_2 _FP32>]);
            $macro_identifier!(f64, f64, [<$graphblas_identifier_1 _FP64>], [<$graphblas_identifier_2 _FP64>]);
            graphblas_sparse_linear_algebra_proc_macros::implement_1_type_macro_for_isize_and_2_typed_graphblas_functions_with_implementation_type!($macro_identifier, $graphblas_identifier_1, $graphblas_identifier_2);
            graphblas_sparse_linear_algebra_proc_macros::implement_1_type_macro_for_usize_and_2_typed_graphblas_functions_with_implementation_type!($macro_identifier, $graphblas_identifier_1, $graphblas_identifier_2);
        }
    };
}
pub(crate) use implement_1_type_macro_for_all_value_types_and_2_typed_graphblas_functions_with_implementation_type;

macro_rules! implement_1_type_macro_for_all_value_types_and_4_typed_graphblas_functions_with_implementation_type {
    ($macro_identifier:ident, $graphblas_identifier_1:ident, $graphblas_identifier_2:ident, $graphblas_identifier_3:ident, $graphblas_identifier_4:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, [<$graphblas_identifier_1 _BOOL>], [<$graphblas_identifier_2 _BOOL>], [<$graphblas_identifier_3 _BOOL>], [<$graphblas_identifier_4 _BOOL>]);
            $macro_identifier!(i8, i8, [<$graphblas_identifier_1 _INT8>], [<$graphblas_identifier_2 _INT8>], [<$graphblas_identifier_3 _INT8>], [<$graphblas_identifier_4 _INT8>]);
            $macro_identifier!(i16, i16, [<$graphblas_identifier_1 _INT16>], [<$graphblas_identifier_2 _INT16>], [<$graphblas_identifier_3 _INT16>], [<$graphblas_identifier_4 _INT16>]);
            $macro_identifier!(i32, i32, [<$graphblas_identifier_1 _INT32>], [<$graphblas_identifier_2 _INT32>], [<$graphblas_identifier_3 _INT32>], [<$graphblas_identifier_4 _INT32>]);
            $macro_identifier!(i64, i64, [<$graphblas_identifier_1 _INT64>], [<$graphblas_identifier_2 _INT64>], [<$graphblas_identifier_3 _INT64>], [<$graphblas_identifier_4 _INT64>]);
            $macro_identifier!(u8, u8, [<$graphblas_identifier_1 _UINT8>], [<$graphblas_identifier_2 _UINT8>], [<$graphblas_identifier_3 _UINT8>], [<$graphblas_identifier_4 _UINT8>]);
            $macro_identifier!(u16, u16, [<$graphblas_identifier_1 _UINT16>], [<$graphblas_identifier_2 _UINT16>], [<$graphblas_identifier_3 _UINT16>], [<$graphblas_identifier_4 _UINT16>]);
            $macro_identifier!(u32, u32, [<$graphblas_identifier_1 _UINT32>], [<$graphblas_identifier_2 _UINT32>], [<$graphblas_identifier_3 _UINT32>], [<$graphblas_identifier_4 _UINT32>]);
            $macro_identifier!(u64, u64, [<$graphblas_identifier_1 _UINT64>], [<$graphblas_identifier_2 _UINT64>], [<$graphblas_identifier_3 _UINT64>], [<$graphblas_identifier_4 _UINT64>]);
            $macro_identifier!(f32, f32, [<$graphblas_identifier_1 _FP32>], [<$graphblas_identifier_2 _FP32>], [<$graphblas_identifier_3 _FP32>], [<$graphblas_identifier_4 _FP32>]);
            $macro_identifier!(f64, f64, [<$graphblas_identifier_1 _FP64>], [<$graphblas_identifier_2 _FP64>], [<$graphblas_identifier_3 _FP64>], [<$graphblas_identifier_4 _FP64>]);
            graphblas_sparse_linear_algebra_proc_macros::implement_1_type_macro_for_isize_and_4_typed_graphblas_functions_with_implementation_type!($macro_identifier, $graphblas_identifier_1, $graphblas_identifier_2, $graphblas_identifier_3, $graphblas_identifier_4);
            graphblas_sparse_linear_algebra_proc_macros::implement_1_type_macro_for_usize_and_4_typed_graphblas_functions_with_implementation_type!($macro_identifier, $graphblas_identifier_1, $graphblas_identifier_2, $graphblas_identifier_3, $graphblas_identifier_4);
        }
    };
}
pub(crate) use implement_1_type_macro_for_all_value_types_and_4_typed_graphblas_functions_with_implementation_type;

macro_rules! implement_2_type_macro_for_all_value_types_and_typed_graphblas_function {
    ($macro_identifier:ident, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, [<$graphblas_identifier _BOOL>]);
            $macro_identifier!(i8, i8, [<$graphblas_identifier _INT8>]);
            $macro_identifier!(i16, i16, [<$graphblas_identifier _INT16>]);
            $macro_identifier!(i32, i32, [<$graphblas_identifier _INT32>]);
            $macro_identifier!(i64, i64, [<$graphblas_identifier _INT64>]);
            $macro_identifier!(u8, u8, [<$graphblas_identifier _UINT8>]);
            $macro_identifier!(u16, u16, [<$graphblas_identifier _UINT16>]);
            $macro_identifier!(u32, u32, [<$graphblas_identifier _UINT32>]);
            $macro_identifier!(u64, u64, [<$graphblas_identifier _UINT64>]);
            $macro_identifier!(f32, f32, [<$graphblas_identifier _FP32>]);
            $macro_identifier!(f64, f64, [<$graphblas_identifier _FP64>]);
            // $macro_identifier!(isize, isize, id_isize!($graphblas_identifier));
            // $macro_identifier!(usize, usize, id_usize!($graphblas_identifier));
            graphblas_sparse_linear_algebra_proc_macros::implement_2_type_macro_for_isize_and_typed_graphblas_function!($macro_identifier, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_2_type_macro_for_usize_and_typed_graphblas_function!($macro_identifier, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_2_type_macro_for_all_value_types_and_typed_graphblas_function;

macro_rules! implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!($trait, [<$graphblas_identifier _BOOL>], bool);
            $macro_identifier!($trait, [<$graphblas_identifier _INT8>], i8);
            $macro_identifier!($trait, [<$graphblas_identifier _INT16>], i16);
            $macro_identifier!($trait, [<$graphblas_identifier _INT32>], i32);
            $macro_identifier!($trait, [<$graphblas_identifier _INT64>], i64);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT8>], u8);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT16>], u16);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT32>], u32);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT64>], u64);
            $macro_identifier!($trait, [<$graphblas_identifier _FP32>], f32);
            $macro_identifier!($trait, [<$graphblas_identifier _FP64>], f64);
            // $macro_identifier!($trait, id_isize!($graphblas_identifier), isize);
            // $macro_identifier!($trait, id_usize!($graphblas_identifier), usize);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize!($macro_identifier, $trait, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize!($macro_identifier, $trait, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types;

macro_rules! implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident) => {
        paste::paste! {
            // $macro_identifier!($trait, [<$graphblas_identifier _BOOL>], bool);
            $macro_identifier!($trait, [<$graphblas_identifier _INT8>], i8);
            $macro_identifier!($trait, [<$graphblas_identifier _INT16>], i16);
            $macro_identifier!($trait, [<$graphblas_identifier _INT32>], i32);
            $macro_identifier!($trait, [<$graphblas_identifier _INT64>], i64);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT8>], u8);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT16>], u16);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT32>], u32);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT64>], u64);
            // $macro_identifier!($trait, [<$graphblas_identifier _FP32>], f32);
            // $macro_identifier!($trait, [<$graphblas_identifier _FP64>], f64);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize!($macro_identifier, $trait, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize!($macro_identifier, $trait, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types;

macro_rules! implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident) => {
        paste::paste! {
            // $macro_identifier!($trait, [<$graphblas_identifier _BOOL>], bool);
            // $macro_identifier!($trait, [<$graphblas_identifier _INT8>], i8);
            // $macro_identifier!($trait, [<$graphblas_identifier _INT16>], i16);
            $macro_identifier!($trait, [<$graphblas_identifier _INT32>], i32);
            $macro_identifier!($trait, [<$graphblas_identifier _INT64>], i64);
            // $macro_identifier!($trait, [<$graphblas_identifier _UINT8>], u8);
            // $macro_identifier!($trait, [<$graphblas_identifier _UINT16>], u16);
            // $macro_identifier!($trait, [<$graphblas_identifier _UINT32>], u32);
            // $macro_identifier!($trait, [<$graphblas_identifier _UINT64>], u64);
            // $macro_identifier!($trait, [<$graphblas_identifier _FP32>], f32);
            // $macro_identifier!($trait, [<$graphblas_identifier _FP64>], f64);
            // graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize!($macro_identifier, $trait, $graphblas_identifier);
            // graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize!($macro_identifier, $trait, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types;

macro_rules! implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!($trait, [<$graphblas_identifier _FP32>], f32);
            $macro_identifier!($trait, [<$graphblas_identifier _FP64>], f64);
        }
    };
}
pub(crate) use implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types;

macro_rules! implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident) => {
        paste::paste! {
            // $macro_identifier!($trait, [<$graphblas_identifier _BOOL>], bool);
            $macro_identifier!($trait, [<$graphblas_identifier _INT8>], i8);
            $macro_identifier!($trait, [<$graphblas_identifier _INT16>], i16);
            $macro_identifier!($trait, [<$graphblas_identifier _INT32>], i32);
            $macro_identifier!($trait, [<$graphblas_identifier _INT64>], i64);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT8>], u8);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT16>], u16);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT32>], u32);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT64>], u64);
            $macro_identifier!($trait, [<$graphblas_identifier _FP32>], f32);
            $macro_identifier!($trait, [<$graphblas_identifier _FP64>], f64);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize!($macro_identifier, $trait, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize!($macro_identifier, $trait, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool;

macro_rules! implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_with_postfix {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident, $postfix:ident) => {
        paste::paste! {
            $macro_identifier!($trait, [<$graphblas_identifier _BOOL_ $postfix>], bool);
            $macro_identifier!($trait, [<$graphblas_identifier _INT8_ $postfix>], i8);
            $macro_identifier!($trait, [<$graphblas_identifier _INT16_ $postfix>], i16);
            $macro_identifier!($trait, [<$graphblas_identifier _INT32_ $postfix>], i32);
            $macro_identifier!($trait, [<$graphblas_identifier _INT64_ $postfix>], i64);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT8_ $postfix>], u8);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT16_ $postfix>], u16);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT32_ $postfix>], u32);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT64_ $postfix>], u64);
            $macro_identifier!($trait, [<$graphblas_identifier _FP32_ $postfix>], f32);
            $macro_identifier!($trait, [<$graphblas_identifier _FP64_ $postfix>], f64);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize_with_postfix!($macro_identifier, $trait, $graphblas_identifier,  $postfix);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize_with_postfix!($macro_identifier, $trait, $graphblas_identifier,  $postfix);
        }
    };
}
pub(crate) use implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_with_postfix;

macro_rules! implement_macro_with_2_type_trait_and_typed_graphblas_function_for_all_value_types {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!($trait, [<$graphblas_identifier _BOOL>], bool, bool);
            $macro_identifier!($trait, [<$graphblas_identifier _INT8>], i8, i8);
            $macro_identifier!($trait, [<$graphblas_identifier _INT16>], i16, i16);
            $macro_identifier!($trait, [<$graphblas_identifier _INT32>], i32, i32);
            $macro_identifier!($trait, [<$graphblas_identifier _INT64>], i64, i64);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT8>], u8, u8);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT16>], u16, u16);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT32>], u32, u32);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT64>], u64, u64);
            $macro_identifier!($trait, [<$graphblas_identifier _FP32>], f32, f32);
            $macro_identifier!($trait, [<$graphblas_identifier _FP64>], f64, f64);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_2_typed_trait_and_typed_graphblas_function_for_isize!($macro_identifier, $trait, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_2_typed_trait_and_typed_graphblas_function_for_usize!($macro_identifier, $trait, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_macro_with_2_type_trait_and_typed_graphblas_function_for_all_value_types;

macro_rules! implement_macro_with_2_type_trait_and_output_type_and_typed_graphblas_function_for_all_value_types {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident, $output_type: ty) => {
        paste::paste! {
            $macro_identifier!($trait, [<$graphblas_identifier _BOOL>], bool, bool, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _INT8>], i8, i8, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _INT16>], i16, i16, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _INT32>], i32, i32, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _INT64>], i64, i64, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT8>], u8, u8, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT16>], u16, u16, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT32>], u32, u32, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT64>], u64, u64, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _FP32>], f32, f32, $output_type);
            $macro_identifier!($trait, [<$graphblas_identifier _FP64>], f64, f64, $output_type);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_2_typed_trait_and_output_type_and_typed_graphblas_function_for_isize!($macro_identifier, $trait, $graphblas_identifier, $output_type);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_2_typed_trait_and_output_type_and_typed_graphblas_function_for_usize!($macro_identifier, $trait, $graphblas_identifier, $output_type);
        }
    };
}
pub(crate) use implement_macro_with_2_type_trait_and_output_type_and_typed_graphblas_function_for_all_value_types;

macro_rules! implement_macro_with_3_type_trait_and_typed_graphblas_function_for_all_value_types {
    ($macro_identifier:ident, $trait:ty, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!($trait, [<$graphblas_identifier _BOOL>], bool, bool, bool);
            $macro_identifier!($trait, [<$graphblas_identifier _INT8>], i8, i8, i8);
            $macro_identifier!($trait, [<$graphblas_identifier _INT16>], i16, i16, i16);
            $macro_identifier!($trait, [<$graphblas_identifier _INT32>], i32, i32, i32);
            $macro_identifier!($trait, [<$graphblas_identifier _INT64>], i64, i64, i64);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT8>], u8, u8, u8);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT16>], u16, u16, u16);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT32>], u32, u32, u32);
            $macro_identifier!($trait, [<$graphblas_identifier _UINT64>], u64, u64, u64);
            $macro_identifier!($trait, [<$graphblas_identifier _FP32>], f32, f32, f32);
            $macro_identifier!($trait, [<$graphblas_identifier _FP64>], f64, f64, f64);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_3_typed_trait_and_typed_graphblas_function_for_isize!($macro_identifier, $trait, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_with_3_typed_trait_and_typed_graphblas_function_for_usize!($macro_identifier, $trait, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_macro_with_3_type_trait_and_typed_graphblas_function_for_all_value_types;

macro_rules! implement_macro_for_all_value_types_and_graphblas_function {
    ($macro_identifier:ident, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!(bool, [<$graphblas_identifier _BOOL>]);
            $macro_identifier!(i8, [<$graphblas_identifier _INT8>]);
            $macro_identifier!(i16, [<$graphblas_identifier _INT16>]);
            $macro_identifier!(i32, [<$graphblas_identifier _INT32>]);
            $macro_identifier!(i64, [<$graphblas_identifier _INT64>]);
            $macro_identifier!(u8, [<$graphblas_identifier _UINT8>]);
            $macro_identifier!(u16, [<$graphblas_identifier _UINT16>]);
            $macro_identifier!(u32, [<$graphblas_identifier _UINT32>]);
            $macro_identifier!(u64, [<$graphblas_identifier _UINT64>]);
            $macro_identifier!(f32, [<$graphblas_identifier _FP32>]);
            $macro_identifier!(f64, [<$graphblas_identifier _FP64>]);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_for_isize_and_graphblas_function!($macro_identifier, $graphblas_identifier);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_for_usize_and_graphblas_function!($macro_identifier, $graphblas_identifier);
        }
    };
}
pub(crate) use implement_macro_for_all_value_types_and_graphblas_function;

macro_rules! implement_macro_for_all_value_types_and_2_typed_graphblas_functions {
    ($macro_identifier:ident, $graphblas_identifier_1:ident, $graphblas_identifier_2:ident) => {
        paste::paste! {
            $macro_identifier!(bool, [<$graphblas_identifier_1 _BOOL>], [<$graphblas_identifier_2 _BOOL>]);
            $macro_identifier!(i8, [<$graphblas_identifier_1 _INT8>], [<$graphblas_identifier_2 _INT8>]);
            $macro_identifier!(i16, [<$graphblas_identifier_1 _INT16>], [<$graphblas_identifier_2 _INT16>]);
            $macro_identifier!(i32, [<$graphblas_identifier_1 _INT32>], [<$graphblas_identifier_2 _INT32>]);
            $macro_identifier!(i64, [<$graphblas_identifier_1 _INT64>], [<$graphblas_identifier_2 _INT64>]);
            $macro_identifier!(u8, [<$graphblas_identifier_1 _UINT8>], [<$graphblas_identifier_2 _UINT8>]);
            $macro_identifier!(u16, [<$graphblas_identifier_1 _UINT16>], [<$graphblas_identifier_2 _UINT16>]);
            $macro_identifier!(u32, [<$graphblas_identifier_1 _UINT32>], [<$graphblas_identifier_2 _UINT32>]);
            $macro_identifier!(u64, [<$graphblas_identifier_1 _UINT64>], [<$graphblas_identifier_2 _UINT64>]);
            $macro_identifier!(f32, [<$graphblas_identifier_1 _FP32>], [<$graphblas_identifier_2 _FP32>]);
            $macro_identifier!(f64, [<$graphblas_identifier_1 _FP64>], [<$graphblas_identifier_2 _FP64>]);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_for_isize_and_2_typed_graphblas_functions!($macro_identifier, $graphblas_identifier_1, $graphblas_identifier_2);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_for_usize_and_2_typed_graphblas_functions!($macro_identifier, $graphblas_identifier_1, $graphblas_identifier_2);
        }
    };
}
pub(crate) use implement_macro_for_all_value_types_and_2_typed_graphblas_functions;

macro_rules! implement_macro_for_all_value_types_and_2_typed_graphblas_functions_with_mutable_scalar_type_conversion {
    ($macro_identifier:ident, $graphblas_identifier_1:ident, $graphblas_identifier_2:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, [<$graphblas_identifier_1 _BOOL>], [<$graphblas_identifier_2 _BOOL>], identity_conversion);
            $macro_identifier!(i8, i8, [<$graphblas_identifier_1 _INT8>], [<$graphblas_identifier_2 _INT8>], identity_conversion);
            $macro_identifier!(i16, i16, [<$graphblas_identifier_1 _INT16>], [<$graphblas_identifier_2 _INT16>], identity_conversion);
            $macro_identifier!(i32, i32, [<$graphblas_identifier_1 _INT32>], [<$graphblas_identifier_2 _INT32>], identity_conversion);
            $macro_identifier!(i64, i64, [<$graphblas_identifier_1 _INT64>], [<$graphblas_identifier_2 _INT64>], identity_conversion);
            $macro_identifier!(u8, u8, [<$graphblas_identifier_1 _UINT8>], [<$graphblas_identifier_2 _UINT8>], identity_conversion);
            $macro_identifier!(u16, u16, [<$graphblas_identifier_1 _UINT16>], [<$graphblas_identifier_2 _UINT16>], identity_conversion);
            $macro_identifier!(u32, u32, [<$graphblas_identifier_1 _UINT32>], [<$graphblas_identifier_2 _UINT32>], identity_conversion);
            $macro_identifier!(u64, u64, [<$graphblas_identifier_1 _UINT64>], [<$graphblas_identifier_2 _UINT64>], identity_conversion);
            $macro_identifier!(f32, f32, [<$graphblas_identifier_1 _FP32>], [<$graphblas_identifier_2 _FP32>], identity_conversion);
            $macro_identifier!(f64, f64, [<$graphblas_identifier_1 _FP64>], [<$graphblas_identifier_2 _FP64>], identity_conversion);
            // $macro_identifier!(isize, graphblas_sparse_linear_algebra_proc_macros::graphblas_implementation_type_for_isize!(), graphblas_sparse_linear_algebra_proc_macros::graphblas_identifier_for_isize!($graphblas_identifier_1), graphblas_sparse_linear_algebra_proc_macros::graphblas_identifier_for_isize!($graphblas_identifier_2), convert_mut_scalar_to_type);
            // $macro_identifier!(usize, graphblas_sparse_linear_algebra_proc_macros::graphblas_implementation_type_for_usize!(), graphblas_sparse_linear_algebra_proc_macros::graphblas_identifier_for_usize!($graphblas_identifier_1), graphblas_sparse_linear_algebra_proc_macros::graphblas_identifier_for_usize!($graphblas_identifier_2), convert_mut_scalar_to_type);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_for_isize_and_2_typed_graphblas_functions_with_type_conversion!($macro_identifier, $graphblas_identifier_1, $graphblas_identifier_2, convert_mut_scalar_to_type);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_for_usize_and_2_typed_graphblas_functions_with_type_conversion!($macro_identifier, $graphblas_identifier_1, $graphblas_identifier_2, convert_mut_scalar_to_type);
        }
    };
}
pub(crate) use implement_macro_for_all_value_types_and_2_typed_graphblas_functions_with_mutable_scalar_type_conversion;

macro_rules! implement_2_type_macro_for_all_value_types_and_typed_graphblas_function_with_scalar_type_conversion {
    ($macro_identifier:ident, $graphblas_identifier:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, bool, [<$graphblas_identifier _BOOL>], identity_conversion);
            $macro_identifier!(i8, i8, i8, [<$graphblas_identifier _INT8>], identity_conversion);
            $macro_identifier!(i16, i16, i16, [<$graphblas_identifier _INT16>], identity_conversion);
            $macro_identifier!(i32, i32, i32, [<$graphblas_identifier _INT32>], identity_conversion);
            $macro_identifier!(i64, i64, i64, [<$graphblas_identifier _INT64>], identity_conversion);
            $macro_identifier!(u8, u8, u8, [<$graphblas_identifier _UINT8>], identity_conversion);
            $macro_identifier!(u16, u16, u16, [<$graphblas_identifier _UINT16>], identity_conversion);
            $macro_identifier!(u32, u32, u32, [<$graphblas_identifier _UINT32>], identity_conversion);
            $macro_identifier!(u64, u64, u64, [<$graphblas_identifier _UINT64>], identity_conversion);
            $macro_identifier!(f32, f32, f32, [<$graphblas_identifier _FP32>], identity_conversion);
            $macro_identifier!(f64, f64, f64, [<$graphblas_identifier _FP64>], identity_conversion);
            graphblas_sparse_linear_algebra_proc_macros::implement_2_type_macro_for_isize_and_typed_graphblas_function_with_type_conversion!($macro_identifier, $graphblas_identifier, convert_scalar_to_type);
            graphblas_sparse_linear_algebra_proc_macros::implement_2_type_macro_for_usize_and_typed_graphblas_function_with_type_conversion!($macro_identifier, $graphblas_identifier, convert_scalar_to_type);
        }
    };
}
pub(crate) use implement_2_type_macro_for_all_value_types_and_typed_graphblas_function_with_scalar_type_conversion;

// macro_rules! implement_macro_with_3_types_and_4_graphblas_functions_for_all_data_types {
//     ($macro_identifier:ident, $graphblas_function_1:ident, $graphblas_function_2:ident, $graphblas_function_3:ident, $graphblas_function_4:ident) => {
//         paste::paste! {
//             $macro_identifier!(bool, bool, bool, [<$graphblas_function_1 _BOOL>], [<$graphblas_function_2 _BOOL>], [<$graphblas_function_3 _BOOL>], [<$graphblas_function_4 _BOOL>]);
//             $macro_identifier!(i8, i8, i8, [<$graphblas_function_1 _INT8>], [<$graphblas_function_2 _INT8>], [<$graphblas_function_3 _INT8>], [<$graphblas_function_4 _INT8>]);
//             $macro_identifier!(i16, i16, i16, [<$graphblas_function_1 _INT16>], [<$graphblas_function_2 _INT16>], [<$graphblas_function_3 _INT16>], [<$graphblas_function_4 _INT16>]);
//             $macro_identifier!(i32, i32, i32, [<$graphblas_function_1 _INT32>], [<$graphblas_function_2 _INT32>], [<$graphblas_function_3 _INT32>], [<$graphblas_function_4 _INT32>]);
//             $macro_identifier!(i64, i64, i64, [<$graphblas_function_1 _INT64>], [<$graphblas_function_2 _INT64>], [<$graphblas_function_3 _INT64>], [<$graphblas_function_4 _INT64>]);
//             $macro_identifier!(u8, u8, u8, [<$graphblas_function_1 _UINT8>], [<$graphblas_function_2 _UINT8>], [<$graphblas_function_3 _UINT8>], [<$graphblas_function_4 _UINT8>]);
//             $macro_identifier!(u16, u16, u16, [<$graphblas_function_1 _UINT16>], [<$graphblas_function_2 _UINT16>], [<$graphblas_function_3 _UINT16>], [<$graphblas_function_4 _UINT16>]);
//             $macro_identifier!(u32, u32, u32, [<$graphblas_function_1 _UINT32>], [<$graphblas_function_2 _UINT32>], [<$graphblas_function_3 _UINT32>], [<$graphblas_function_4 _UINT32>]);
//             $macro_identifier!(u64, u64, u64, [<$graphblas_function_1 _UINT64>], [<$graphblas_function_2 _UINT64>], [<$graphblas_function_3 _UINT64>], [<$graphblas_function_4 _UINT64>]);
//             $macro_identifier!(f32, f32, f32, [<$graphblas_function_1 _FP32>], [<$graphblas_function_2 _FP32>], [<$graphblas_function_3 _FP32>], [<$graphblas_function_4 _FP32>]);
//             $macro_identifier!(f64, f64, f64, [<$graphblas_function_1 _FP64>], [<$graphblas_function_2 _FP64>], [<$graphblas_function_3 _FP64>], [<$graphblas_function_4 _FP64>]);
//             macros_to_implement_traits::implement_macro_for_3_isizes_and_4_graphblas_functions!($macro_identifier, $graphblas_function_1, $graphblas_function_2, $graphblas_function_3, $graphblas_function_4);
//             macros_to_implement_traits::implement_macro_for_3_usizes_and_4_graphblas_functions!($macro_identifier, $graphblas_function_1, $graphblas_function_2, $graphblas_function_3, $graphblas_function_4);
//         }
//     };
// }
// pub(crate) use implement_macro_with_3_types_and_4_graphblas_functions_for_all_data_types;
macro_rules! implement_macro_with_3_types_and_4_graphblas_functions_with_scalar_conversion_for_all_data_types {
    ($macro_identifier:ident, $graphblas_function_1:ident, $graphblas_function_2:ident, $graphblas_function_3:ident, $graphblas_function_4:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, bool, bool, identity_conversion, [<$graphblas_function_1 _BOOL>], [<$graphblas_function_2 _BOOL>], [<$graphblas_function_3 _BOOL>], [<$graphblas_function_4 _BOOL>]);
            $macro_identifier!(i8, i8, i8, i8, identity_conversion, [<$graphblas_function_1 _INT8>], [<$graphblas_function_2 _INT8>], [<$graphblas_function_3 _INT8>], [<$graphblas_function_4 _INT8>]);
            $macro_identifier!(i16, i16, i16, i16, identity_conversion, [<$graphblas_function_1 _INT16>], [<$graphblas_function_2 _INT16>], [<$graphblas_function_3 _INT16>], [<$graphblas_function_4 _INT16>]);
            $macro_identifier!(i32, i32, i32, i32, identity_conversion, [<$graphblas_function_1 _INT32>], [<$graphblas_function_2 _INT32>], [<$graphblas_function_3 _INT32>], [<$graphblas_function_4 _INT32>]);
            $macro_identifier!(i64, i64, i64, i64, identity_conversion, [<$graphblas_function_1 _INT64>], [<$graphblas_function_2 _INT64>], [<$graphblas_function_3 _INT64>], [<$graphblas_function_4 _INT64>]);
            $macro_identifier!(u8, u8, u8, u8, identity_conversion, [<$graphblas_function_1 _UINT8>], [<$graphblas_function_2 _UINT8>], [<$graphblas_function_3 _UINT8>], [<$graphblas_function_4 _UINT8>]);
            $macro_identifier!(u16, u16, u16, u16, identity_conversion, [<$graphblas_function_1 _UINT16>], [<$graphblas_function_2 _UINT16>], [<$graphblas_function_3 _UINT16>], [<$graphblas_function_4 _UINT16>]);
            $macro_identifier!(u32, u32, u32, u32, identity_conversion, [<$graphblas_function_1 _UINT32>], [<$graphblas_function_2 _UINT32>], [<$graphblas_function_3 _UINT32>], [<$graphblas_function_4 _UINT32>]);
            $macro_identifier!(u64, u64, u64, u64, identity_conversion, [<$graphblas_function_1 _UINT64>], [<$graphblas_function_2 _UINT64>], [<$graphblas_function_3 _UINT64>], [<$graphblas_function_4 _UINT64>]);
            $macro_identifier!(f32, f32, f32, f32, identity_conversion, [<$graphblas_function_1 _FP32>], [<$graphblas_function_2 _FP32>], [<$graphblas_function_3 _FP32>], [<$graphblas_function_4 _FP32>]);
            $macro_identifier!(f64, f64, f64, f64, identity_conversion, [<$graphblas_function_1 _FP64>], [<$graphblas_function_2 _FP64>], [<$graphblas_function_3 _FP64>], [<$graphblas_function_4 _FP64>]);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_for_3_isizes_and_4_graphblas_functions!($macro_identifier, $graphblas_function_1, $graphblas_function_2, $graphblas_function_3, $graphblas_function_4, convert_scalar_to_type);
            graphblas_sparse_linear_algebra_proc_macros::implement_macro_for_3_usizes_and_4_graphblas_functions!($macro_identifier, $graphblas_function_1, $graphblas_function_2, $graphblas_function_3, $graphblas_function_4, convert_scalar_to_type);
        }
    };
}
pub(crate) use implement_macro_with_3_types_and_4_graphblas_functions_with_scalar_conversion_for_all_data_types;

macro_rules! implement_semiring_for_all_value_types {
    ($macro_identifier:ident, $semiring:ident, $addition_operator:ident, $multiplication_operator:ident, $graphblas_operator:ident) => {
        paste::paste! {
            // $macro_identifier!($semiring, $addition_operator, $multiplication_operator, bool, bool, bool, [<$graphblas_operator _BOOL>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, i8, i8, i8, i8, [<$graphblas_operator _INT8>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, i16, i16, i16, i16, [<$graphblas_operator _INT16>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, i32, i32, i32, i32, [<$graphblas_operator _INT32>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, i64, i64, i64, i64, [<$graphblas_operator _INT64>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, u8, u8, u8, u8, [<$graphblas_operator _UINT8>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, u16, u16, u16, u16, [<$graphblas_operator _UINT16>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, u32, u32, u32, u32, [<$graphblas_operator _UINT32>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, u64, u64, u64, u64, [<$graphblas_operator _UINT64>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, f32, f32, f32, f32, [<$graphblas_operator _FP32>]);
            $macro_identifier!($semiring, $addition_operator, $multiplication_operator, f64, f64, f64, f64, [<$graphblas_operator _FP64>]);
            graphblas_sparse_linear_algebra_proc_macros::implement_semiring_for_isize!($macro_identifier, $semiring, $addition_operator, $multiplication_operator, $graphblas_operator);
            graphblas_sparse_linear_algebra_proc_macros::implement_semiring_for_usize!($macro_identifier, $semiring, $addition_operator, $multiplication_operator, $graphblas_operator);
        }
    };
}
pub(crate) use implement_semiring_for_all_value_types;

// macro_rules! scalar_indentity_conversion {
//     ($from_type: ty, $to_type: ty) => {
//         Ok(self)
//     };
// }

// macro_rules! scalar_conversion {
//     ($from_type: ty, $to_type: ty) => {
//         let as_type: Result<i64, std::num::TryFromIntError> = self.try_into();
//         match as_type {
//             Ok(as_type) => Ok(as_type),
//             Err(error) => {
//                 Err(LogicError::from(error).into())
//             }
//         }
//     };
// }

macro_rules! implement_type_conversion_macro {
    ($macro_identifier:ident, $identity_conversion:ident, $conversion_implementation:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, $identity_conversion);
            $macro_identifier!(i8, i8, $identity_conversion);
            $macro_identifier!(i16, i16, $identity_conversion);
            $macro_identifier!(i32, i32, $identity_conversion);
            $macro_identifier!(i64, i64, $identity_conversion);
            $macro_identifier!(u8, u8, $identity_conversion);
            $macro_identifier!(u16, u16, $identity_conversion);
            $macro_identifier!(u32, u32, $identity_conversion);
            $macro_identifier!(u64, u64, $identity_conversion);
            $macro_identifier!(f32, f32, $identity_conversion);
            $macro_identifier!(f64, f64, $identity_conversion);
            graphblas_sparse_linear_algebra_proc_macros::implement_type_conversion_macro_for_isize!($macro_identifier, $conversion_implementation);
            graphblas_sparse_linear_algebra_proc_macros::implement_type_conversion_macro_for_usize!($macro_identifier, $conversion_implementation);
        }
    };
}
pub(crate) use implement_type_conversion_macro;

// TODO: error catching and propagation
macro_rules! convert_scalar_to_type {
    ($variable: ident, $target_type: ty) => {
        let $variable: $target_type = $variable.to_owned().try_into().unwrap();
    };
}
pub(crate) use convert_scalar_to_type;

macro_rules! convert_mut_scalar_to_type {
    ($variable: ident, $target_type: ty) => {
        let mut $variable: $target_type = $variable.to_owned().try_into().unwrap();
    };
}
pub(crate) use convert_mut_scalar_to_type;

macro_rules! identity_conversion {
    ($variable: ident, $_target_type: ty) => {
        // let $variable = std::convert::identity($variable);
    };
}
pub(crate) use identity_conversion;
