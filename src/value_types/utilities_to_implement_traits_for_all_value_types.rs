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

macro_rules! implement_macro_for_all_value_types_and_graphblas_function {
    ($macro_identifier:ident, $graphblas_indentifier:ident) => {
        paste::paste! {
            $macro_identifier!(bool, [<$graphblas_indentifier _BOOL>]);
            $macro_identifier!(i8, [<$graphblas_indentifier _INT8>]);
            $macro_identifier!(i16, [<$graphblas_indentifier _INT16>]);
            $macro_identifier!(i32, [<$graphblas_indentifier _INT32>]);
            $macro_identifier!(i64, [<$graphblas_indentifier _INT64>]);
            $macro_identifier!(u8, [<$graphblas_indentifier _UINT8>]);
            $macro_identifier!(u16, [<$graphblas_indentifier _UINT16>]);
            $macro_identifier!(u32, [<$graphblas_indentifier _UINT32>]);
            $macro_identifier!(u64, [<$graphblas_indentifier _UINT64>]);
            $macro_identifier!(f32, [<$graphblas_indentifier _FP32>]);
            $macro_identifier!(f64, [<$graphblas_indentifier _FP64>]);
            macros_to_implement_traits::implement_macro_for_isize_and_graphblas_function!($macro_identifier, $graphblas_indentifier);
            macros_to_implement_traits::implement_macro_for_usize_and_graphblas_function!($macro_identifier, $graphblas_indentifier);
        }
    };
}
pub(crate) use implement_macro_for_all_value_types_and_graphblas_function;

macro_rules! implement_macro_for_all_value_types_and_graphblas_function_with_vector_type_conversion {
    ($macro_identifier:ident, $graphblas_indentifier:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, [<$graphblas_indentifier _BOOL>], identity_conversion);
            $macro_identifier!(i8, i8, [<$graphblas_indentifier _INT8>], identity_conversion);
            $macro_identifier!(i16, i16, [<$graphblas_indentifier _INT16>], identity_conversion);
            $macro_identifier!(i32, i32, [<$graphblas_indentifier _INT32>], identity_conversion);
            $macro_identifier!(i64, i64, [<$graphblas_indentifier _INT64>], identity_conversion);
            $macro_identifier!(u8, u8, [<$graphblas_indentifier _UINT8>], identity_conversion);
            $macro_identifier!(u16, u16, [<$graphblas_indentifier _UINT16>], identity_conversion);
            $macro_identifier!(u32, u32, [<$graphblas_indentifier _UINT32>], identity_conversion);
            $macro_identifier!(u64, u64, [<$graphblas_indentifier _UINT64>], identity_conversion);
            $macro_identifier!(f32, f32, [<$graphblas_indentifier _FP32>], identity_conversion);
            $macro_identifier!(f64, f64, [<$graphblas_indentifier _FP64>], identity_conversion);
            macros_to_implement_traits::implement_macro_for_isize_and_graphblas_function_with_type_conversion!($macro_identifier, $graphblas_indentifier, convert_vector_to_type);
            macros_to_implement_traits::implement_macro_for_usize_and_graphblas_function_with_type_conversion!($macro_identifier, $graphblas_indentifier, convert_vector_to_type);
        }
    };
}
pub(crate) use implement_macro_for_all_value_types_and_graphblas_function_with_vector_type_conversion;

macro_rules! implement_macro_for_all_value_types_and_graphblas_function_with_scalar_type_conversion {
    ($macro_identifier:ident, $graphblas_indentifier:ident) => {
        paste::paste! {
            $macro_identifier!(bool, bool, [<$graphblas_indentifier _BOOL>], identity_conversion);
            $macro_identifier!(i8, i8, [<$graphblas_indentifier _INT8>], identity_conversion);
            $macro_identifier!(i16, i16, [<$graphblas_indentifier _INT16>], identity_conversion);
            $macro_identifier!(i32, i32, [<$graphblas_indentifier _INT32>], identity_conversion);
            $macro_identifier!(i64, i64, [<$graphblas_indentifier _INT64>], identity_conversion);
            $macro_identifier!(u8, u8, [<$graphblas_indentifier _UINT8>], identity_conversion);
            $macro_identifier!(u16, u16, [<$graphblas_indentifier _UINT16>], identity_conversion);
            $macro_identifier!(u32, u32, [<$graphblas_indentifier _UINT32>], identity_conversion);
            $macro_identifier!(u64, u64, [<$graphblas_indentifier _UINT64>], identity_conversion);
            $macro_identifier!(f32, f32, [<$graphblas_indentifier _FP32>], identity_conversion);
            $macro_identifier!(f64, f64, [<$graphblas_indentifier _FP64>], identity_conversion);
            macros_to_implement_traits::implement_macro_for_isize_and_graphblas_function_with_type_conversion!($macro_identifier, $graphblas_indentifier, convert_scalar_to_type);
            macros_to_implement_traits::implement_macro_for_usize_and_graphblas_function_with_type_conversion!($macro_identifier, $graphblas_indentifier, convert_scalar_to_type);
        }
    };
}
pub(crate) use implement_macro_for_all_value_types_and_graphblas_function_with_scalar_type_conversion;

macro_rules! convert_vector_to_type {
    ($input: ident, $output: ident, $target_type: ty) => {
        let $output: Vec<$target_type> = $input
            .to_owned()
            .into_par_iter()
            .map(|x| x.try_into().unwrap())
            .collect();
    };
}
pub(crate) use convert_vector_to_type;

// TODO: error catching and propagation
macro_rules! convert_scalar_to_type {
    ($input: ident, $output: ident, $target_type: ty) => {
        let $output: $target_type = $input.to_owned().try_into().unwrap();
    };
}
pub(crate) use convert_scalar_to_type;

macro_rules! identity_conversion {
    ($input: ident, $output: ident, $_target_type: ty) => {
        let $output = std::convert::identity($input);
    };
}
pub(crate) use identity_conversion;
