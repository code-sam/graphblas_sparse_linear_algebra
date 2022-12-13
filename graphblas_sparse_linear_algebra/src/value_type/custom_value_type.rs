use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::bindings_to_graphblas_implementation::{
    GrB_BOOL, GrB_FP32, GrB_FP64, GrB_INT16, GrB_INT32, GrB_INT64, GrB_INT8, GrB_Index, GrB_Type,
    GrB_Type_free, GrB_UINT16, GrB_UINT32, GrB_UINT64, GrB_UINT8, GxB_Type_size,
};
use crate::context::{CallGraphBlasContext, Context};
use crate::error::{SparseLinearAlgebraError, SystemError, SystemErrorType};
use crate::index::{ElementIndex, IndexConversion};

use super::ValueType;

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
        let _ = context.call(
            || unsafe { GrB_Type_free(&mut self.graphblas_type.clone()) },
            &self.graphblas_type,
        );
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

        context.call(
            || unsafe { GxB_Type_size(size.as_mut_ptr(), self.to_graphblas_type()) },
            &self.to_graphblas_type(),
        )?;

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

                context.call_without_detailed_error_information(|| unsafe {
                    $crate::bindings_to_graphblas_implementation::GrB_Type_new(
                        graphblas_type.as_mut_ptr(),
                        size_of_self,
                    )
                })?;

                let graphblas_type = unsafe { graphblas_type.assume_init() };

                Ok(Arc::new(
                    $crate::value_type::RegisteredCustomValueType::new(
                        context,
                        graphblas_type,
                    ),
                ))
            }
        }
    };
}

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
