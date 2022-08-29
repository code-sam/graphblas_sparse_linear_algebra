use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;
// use std::sync::{Arc, Mutex};

// use once_cell::sync::Lazy;

use crate::bindings_to_graphblas_implementation::{
    GrB_BOOL, GrB_FP32, GrB_FP64, GrB_INT16, GrB_INT32, GrB_INT64, GrB_INT8, GrB_Index, GrB_Type,
    GrB_Type_free, GrB_UINT16, GrB_UINT32, GrB_UINT64, GrB_UINT8, GxB_Type_size,
};
use crate::context::Context;
use crate::error::{SparseLinearAlgebraError, SystemError, SystemErrorType};
use crate::util::{ElementIndex, IndexConversion};

// use macros_to_implement_traits::{graphblas_built_in_type_for_isize, graphblas_built_in_type_for_usize};
use macros_to_implement_traits::{implement_macro_for_isize, implement_macro_for_usize};

pub trait ValueType {}

pub trait BuiltInValueType<T> {
    fn to_graphblas_type() -> GrB_Type;
}

pub(crate) trait CustomValueType: ValueType {
    type Type;

    // TODO: what to do when the same type is registered multiple times?
    fn register(
        context: Arc<Context>,
    ) -> Result<Arc<RegisteredCustomValueType<Self::Type>>, SparseLinearAlgebraError>;
}

pub(crate) trait RegisteredValueType<T: ValueType>: ValueType {
    fn to_graphblas_type(registered_value_type: RegisteredCustomValueType<T>) -> GrB_Type;
}

pub(crate) struct RegisteredCustomValueType<T> {
    context: Arc<Context>,
    graphblas_type: GrB_Type,
    _rust_type: PhantomData<T>,
}

impl<T> Drop for RegisteredCustomValueType<T> {
    fn drop(&mut self) {
        let context = self.context.clone();
        let _ = context.call(|| unsafe { GrB_Type_free(&mut self.graphblas_type.clone()) });
    }
}

impl<T> RegisteredCustomValueType<T> {
    pub fn new(context: Arc<Context>, graphblas_type: GrB_Type) -> Self {
        RegisteredCustomValueType {
            context,
            graphblas_type,
            _rust_type: PhantomData,
        }
    }
}

impl<T> RegisteredCustomValueType<T> {
    pub fn to_graphblas_type(&self) -> GrB_Type {
        self.graphblas_type.clone()
    }

    pub fn size_in_graphblas(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let context = self.context.clone();

        let mut size: MaybeUninit<GrB_Index> = MaybeUninit::uninit();

        context.call(|| unsafe { GxB_Type_size(size.as_mut_ptr(), self.to_graphblas_type()) })?;

        let size = unsafe { size.assume_init() };
        Ok(ElementIndex::from_graphblas_index(size)?)
    }

    pub fn context(&self) -> Arc<Context> {
        self.context.clone()
    }
}

// TODO: consider to use a generic type like CustomType<T>(T). This should enable generic trait implementation for all custom types.
// #[macro_export]
macro_rules! implement_value_type_for_custom_type {
    ($value_type: ty) => {
        impl ValueType for $value_type {}

        impl RegisteredValueType<$value_type> for $value_type {
            fn to_graphblas_type(
                registered_value_type: RegisteredCustomValueType<$value_type>,
            ) -> GrB_Type {
                registered_value_type.graphblas_type.clone()
            }
        }

        impl CustomValueType for $value_type {
            type Type = $value_type;

            fn register(
                context: Arc<Context>,
            ) -> Result<Arc<RegisteredCustomValueType<Self::Type>>, SparseLinearAlgebraError> {
                let context = context.clone();
                let mut graphblas_type: MaybeUninit<
                    $crate::bindings_to_graphblas_implementation::GrB_Type,
                > = MaybeUninit::uninit();
                let size_of_self =
                    IndexConversion::as_graphblas_index(std::mem::size_of::<$value_type>())?;

                context.call(|| unsafe {
                    $crate::bindings_to_graphblas_implementation::GrB_Type_new(
                        graphblas_type.as_mut_ptr(),
                        size_of_self,
                    )
                })?;

                let graphblas_type = unsafe { graphblas_type.assume_init() };

                Ok(Arc::new(
                    $crate::value_types::value_type::RegisteredCustomValueType::new(
                        context,
                        graphblas_type,
                    ),
                ))
            }
        }
    };
}

macro_rules! implement_value_type_for_graphblas_built_in_type {
    ($value_type: ty, $graphblas_type_identifier: ident) => {
        impl ValueType for $value_type {}

        impl BuiltInValueType<$value_type> for $value_type {
            fn to_graphblas_type() -> GrB_Type {
                unsafe { $graphblas_type_identifier }
            }
        }
    };
}

// macro_rules! value_type_to_graphblas_type {
//     ($value_type: ty) => {
//         match $value_type {
//             bool => GrB_BOOL,
//             i8 => GrB_INT8,
//             i16 => GrB_INT16,
//             i32 => GrB_INT32,
//             i64 => GrB_INT64,
//             u8 => GrB_UINT8,
//             u16 => GrB_UINT16,
//             u32 => GrB_UINT32,
//             u64 => GrB_UINT64,
//             isize => graphblas_built_in_type_for_isize(), // does calling an equivalent procedural macro improve performacnce by reducing runtime cost?
//             usize => graphblas_built_in_type_for_usize(),
//             _ => panic!("Unsupported type: {:?}", $value_type)
//         }
//     };
// }

// implement_value_type_for_graphblas_built_in_type!(bool, GrB_BOOL);
// implement_value_type_for_graphblas_built_in_type!(i8, GrB_INT8);
// implement_value_type_for_graphblas_built_in_type!(i16, GrB_INT16);
// implement_value_type_for_graphblas_built_in_type!(i32, GrB_INT32);
// implement_value_type_for_graphblas_built_in_type!(i64, GrB_INT64);
// implement_value_type_for_graphblas_built_in_type!(u8, GrB_UINT8);
// implement_value_type_for_graphblas_built_in_type!(u16, GrB_UINT16);
// implement_value_type_for_graphblas_built_in_type!(u32, GrB_UINT32);
// implement_value_type_for_graphblas_built_in_type!(u64, GrB_UINT64);
// implement_value_type_for_graphblas_built_in_type!(f32, GrB_FP32);
// implement_value_type_for_graphblas_built_in_type!(f64, GrB_FP64);
// implement_value_type_for_graphblas_built_in_type!(isize, graphblas_built_in_type_for_isize().unwrap());
// implement_value_type_for_graphblas_built_in_type!(usize, graphblas_built_in_type_for_usize().unwrap());

// pub(crate) fn graphblas_built_in_type_for_usize() -> Result<GrB_Type, SparseLinearAlgebraError> {
//     match usize::BITS {
//         8 => unsafe { Ok(GrB_UINT8) },
//         16 => unsafe { Ok(GrB_UINT16) },
//         32 => unsafe { Ok(GrB_UINT32) },
//         64 => unsafe { Ok(GrB_UINT64) },
//         _ => Err(SystemError::new(
//             SystemErrorType::UnsupportedArchitecture,
//             format!("Unsupported architecture: {} bits", usize::BITS),
//             None,
//         )
//         .into()),
//     }
// }

// pub(crate) fn graphblas_built_in_type_for_isize() -> Result<GrB_Type, SparseLinearAlgebraError> {
//     match usize::BITS {
//         8 => unsafe { Ok(GrB_INT8) },
//         16 => unsafe { Ok(GrB_INT16) },
//         32 => unsafe { Ok(GrB_INT32) },
//         64 => unsafe { Ok(GrB_INT64) },
//         _ => Err(SystemError::new(
//             SystemErrorType::UnsupportedArchitecture,
//             format!("Unsupported architecture: {} bits", usize::BITS),
//             None,
//         )
//         .into()),
//     }
// }

// macro_rules! supported_types {
//     () => {
//         bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, isize, usize,
//     };
// }

macro_rules! implement_graphblas_trait {
    ($macro_identifier:ident) => {
        $macro_identifier!(bool, GrB_BOOL);
        $macro_identifier!(i8, GrB_INT8);
        $macro_identifier!(i16, GrB_INT16);
        $macro_identifier!(i32, GrB_INT32);
        $macro_identifier!(i64, GrB_INT64);
        $macro_identifier!(u8, GrB_UINT8);
        $macro_identifier!(u16, GrB_UINT16);
        $macro_identifier!(u32, GrB_UINT32);
        $macro_identifier!(u64, GrB_UINT64);
        $macro_identifier!(f32, GrB_FP32);
        $macro_identifier!(f64, GrB_FP64);
        implement_macro_for_isize!($macro_identifier);
        implement_macro_for_usize!($macro_identifier);
    };
    // todo: add more arms
}

implement_graphblas_trait!(implement_value_type_for_graphblas_built_in_type);

// impl BuiltInValueType<usize> for usize {
//     fn to_graphblas_type() -> GrB_Type {
//         graphblas_built_in_type_for_usize().unwrap()
//     }
// }
// impl BuiltInValueType<isize> for isize {
//     fn to_graphblas_type() -> GrB_Type {
//         graphblas_built_in_type_for_isize().unwrap()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::Mode;

    #[test]
    fn create_new_custom_type() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        #[repr(C)]
        #[derive(Clone, Debug, PartialEq, PartialOrd)]
        struct CustomType {
            value: String,
            another_value: u128,
        }

        implement_value_type_for_custom_type!(CustomType);

        let custom_type = CustomType::register(context.clone()).unwrap();
        let expected_size = std::mem::size_of::<CustomType>();
        assert_eq!(expected_size, custom_type.size_in_graphblas().unwrap());
    }

    // #[test]
    // fn create_new_type() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();

    //     let built_in_type = bool::register(context.clone()).unwrap();
    //     let expected_size = mem::size_of::<bool>();
    //     assert_eq!(expected_size, built_in_type.size_in_graphblas().unwrap());
    // }

}
