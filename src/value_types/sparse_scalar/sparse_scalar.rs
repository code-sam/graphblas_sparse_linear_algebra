use std::marker::PhantomData;
use std::mem::MaybeUninit;

use std::sync::Arc;

use crate::error::{
    GraphBlasErrorType, LogicErrorType, SparseLinearAlgebraError, SparseLinearAlgebraErrorType,
};

use crate::bindings_to_graphblas_implementation::{
    GrB_Index, GxB_Scalar, GxB_Scalar_clear, GxB_Scalar_dup, GxB_Scalar_extractElement_BOOL,
    GxB_Scalar_extractElement_FP32, GxB_Scalar_extractElement_FP64,
    GxB_Scalar_extractElement_INT16, GxB_Scalar_extractElement_INT32,
    GxB_Scalar_extractElement_INT64, GxB_Scalar_extractElement_INT8,
    GxB_Scalar_extractElement_UINT16, GxB_Scalar_extractElement_UINT32,
    GxB_Scalar_extractElement_UINT64, GxB_Scalar_extractElement_UINT8, GxB_Scalar_free,
    GxB_Scalar_new, GxB_Scalar_nvals, GxB_Scalar_setElement_BOOL, GxB_Scalar_setElement_FP32,
    GxB_Scalar_setElement_FP64, GxB_Scalar_setElement_INT16, GxB_Scalar_setElement_INT32,
    GxB_Scalar_setElement_INT64, GxB_Scalar_setElement_INT8, GxB_Scalar_setElement_UINT16,
    GxB_Scalar_setElement_UINT32, GxB_Scalar_setElement_UINT64, GxB_Scalar_setElement_UINT8,
};
use crate::context::Context;

use crate::util::{ElementIndex, IndexConversion};
use crate::value_types::value_type::{BuiltInValueType, CustomValueType, RegisteredCustomValueType, ValueType};

pub struct SparseScalar<T: ValueType> {
    context: Arc<Context>,
    scalar: GxB_Scalar,
    value_type: PhantomData<T>,
}

// Send and Sync implementaioms should be ok, since mutable access to GxB_Scalar 
// must occur through a mut SparseScalar. Method providing a copy or reference to 
// GxB_Scalar will result in undefined behaviour though. Code review must consider this.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
// unsafe impl Send for SparseScalar<bool> {}
// unsafe impl Send for SparseScalar<u8> {}
// unsafe impl Send for SparseScalar<u16> {}
// unsafe impl Send for SparseScalar<u32> {}
// unsafe impl Send for SparseScalar<u64> {}
// unsafe impl Send for SparseScalar<i8> {}
// unsafe impl Send for SparseScalar<i16> {}
// unsafe impl Send for SparseScalar<i32> {}
// unsafe impl Send for SparseScalar<i64> {}
// unsafe impl Send for SparseScalar<f32> {}
// unsafe impl Send for SparseScalar<f64> {}

// unsafe impl Sync for SparseScalar<bool> {}
// unsafe impl Sync for SparseScalar<u8> {}
// unsafe impl Sync for SparseScalar<u16> {}
// unsafe impl Sync for SparseScalar<u32> {}
// unsafe impl Sync for SparseScalar<u64> {}
// unsafe impl Sync for SparseScalar<i8> {}
// unsafe impl Sync for SparseScalar<i16> {}
// unsafe impl Sync for SparseScalar<i32> {}
// unsafe impl Sync for SparseScalar<i64> {}
// unsafe impl Sync for SparseScalar<f32> {}
// unsafe impl Sync for SparseScalar<f64> {}

impl<T: ValueType + BuiltInValueType<T>> SparseScalar<T> {
    pub fn new(context: &Arc<Context>) -> Result<Self, SparseLinearAlgebraError> {
        let mut scalar: MaybeUninit<GxB_Scalar> = MaybeUninit::uninit();
        let context = context.clone();

        context
            .call(|| unsafe { GxB_Scalar_new(scalar.as_mut_ptr(), <T>::to_graphblas_type()) })?;

        let scalar = unsafe { scalar.assume_init() };
        return Ok(SparseScalar {
            context,
            scalar,
            value_type: PhantomData,
        });
    }
}

impl<T: ValueType + CustomValueType> SparseScalar<T> {
    pub fn new_custom_type(
        value_type: Arc<RegisteredCustomValueType<T>>,
    ) -> Result<Self, SparseLinearAlgebraError> {
        let mut scalar: MaybeUninit<GxB_Scalar> = MaybeUninit::uninit();
        let context = value_type.context();

        context.call(|| unsafe {
            GxB_Scalar_new(scalar.as_mut_ptr(), value_type.to_graphblas_type())
        })?;

        let scalar = unsafe { scalar.assume_init() };
        return Ok(SparseScalar {
            context,
            scalar,
            value_type: PhantomData,
        });
    }
}

impl<T: ValueType> SparseScalar<T> {
    pub fn number_of_stored_elements(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut number_of_values: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context
            .call(|| unsafe { GxB_Scalar_nvals(number_of_values.as_mut_ptr(), self.scalar) })?;
        let number_of_values = unsafe { number_of_values.assume_init() };
        Ok(ElementIndex::from_graphblas_index(number_of_values)?)
    }

    pub fn clear(&mut self) -> Result<(), SparseLinearAlgebraError> {
        self.context
            .call(|| unsafe { GxB_Scalar_clear(self.scalar) })?;
        Ok(())
    }

    pub fn context(&self) -> Arc<Context> {
        self.context.clone()
    }

    pub(crate) fn graphblas_scalar(&self) -> GxB_Scalar {
        self.scalar.clone()
    }
}

impl<T: ValueType> Drop for SparseScalar<T> {
    fn drop(&mut self) -> () {
        let _ = self
            .context
            .call(|| unsafe { GxB_Scalar_free(&mut self.scalar.clone()) });
    }
}

impl<T: ValueType> Clone for SparseScalar<T> {
    fn clone(&self) -> Self {
        let mut scalar_copy: MaybeUninit<GxB_Scalar> = MaybeUninit::uninit();
        self.context
            .call(|| unsafe { GxB_Scalar_dup(scalar_copy.as_mut_ptr(), self.scalar) })
            .unwrap();

        SparseScalar {
            context: self.context.clone(),
            scalar: unsafe { scalar_copy.assume_init() },
            value_type: PhantomData,
        }
    }
}

// TODO improve printing format
// summary data, column aligning
macro_rules! implement_dispay {
    ($value_type:ty) => {
        impl std::fmt::Display for SparseScalar<$value_type> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let value: $value_type;
                match self.get_value() {
                    Err(_error) => return Err(std::fmt::Error),
                    Ok(inner_value) => {
                        value = inner_value;
                    }
                }
                writeln! {f,"Number of stored elements: {:?}", self.number_of_stored_elements()?};
                writeln! {f,"Value: {:?}", value};
                writeln!(f, "")
            }
        }
    };
}

implement_dispay!(bool);
implement_dispay!(i8);
implement_dispay!(i16);
implement_dispay!(i32);
implement_dispay!(i64);
implement_dispay!(u8);
implement_dispay!(u16);
implement_dispay!(u32);
implement_dispay!(u64);
implement_dispay!(f32);
implement_dispay!(f64);

pub trait SetScalarValue<T: ValueType> {
    fn set_value(&mut self, value: &T) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_set_value_for_built_in_type {
    ($value_type:ty, $add_element_function:ident) => {
        impl SetScalarValue<$value_type> for SparseScalar<$value_type> {
            fn set_value(&mut self, value: &$value_type) -> Result<(), SparseLinearAlgebraError> {
                self.context
                    .call(|| unsafe { $add_element_function(self.scalar, *value) })?;
                Ok(())
            }
        }
    };
}

implement_set_value_for_built_in_type!(bool, GxB_Scalar_setElement_BOOL);
implement_set_value_for_built_in_type!(i8, GxB_Scalar_setElement_INT8);
implement_set_value_for_built_in_type!(i16, GxB_Scalar_setElement_INT16);
implement_set_value_for_built_in_type!(i32, GxB_Scalar_setElement_INT32);
implement_set_value_for_built_in_type!(i64, GxB_Scalar_setElement_INT64);
implement_set_value_for_built_in_type!(u8, GxB_Scalar_setElement_UINT8);
implement_set_value_for_built_in_type!(u16, GxB_Scalar_setElement_UINT16);
implement_set_value_for_built_in_type!(u32, GxB_Scalar_setElement_UINT32);
implement_set_value_for_built_in_type!(u64, GxB_Scalar_setElement_UINT64);
implement_set_value_for_built_in_type!(f32, GxB_Scalar_setElement_FP32);
implement_set_value_for_built_in_type!(f64, GxB_Scalar_setElement_FP64);

pub trait GetScalarValue<T: ValueType + Default> {
    fn get_value(&self) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_value_for_built_in_type {
    ($value_type:ty, $get_value_function:ident) => {
        impl GetScalarValue<$value_type> for SparseScalar<$value_type> {
            fn get_value(&self) -> Result<$value_type, SparseLinearAlgebraError> {
                let mut value: MaybeUninit<$value_type> = MaybeUninit::uninit();

                let result = self
                    .context
                    .call(|| unsafe { $get_value_function(value.as_mut_ptr(), self.scalar) });

                match result {
                    Ok(_) => {
                        let value = unsafe { value.assume_init() };
                        Ok(value)
                    }
                    Err(error) => match error.error_type() {
                        SparseLinearAlgebraErrorType::LogicErrorType(
                            LogicErrorType::GraphBlas(GraphBlasErrorType::NoValue),
                        ) => Ok(<$value_type>::default()),
                        _ => Err(error),
                    },
                }
            }
        }
    };
}

implement_get_value_for_built_in_type!(bool, GxB_Scalar_extractElement_BOOL);
implement_get_value_for_built_in_type!(i8, GxB_Scalar_extractElement_INT8);
implement_get_value_for_built_in_type!(i16, GxB_Scalar_extractElement_INT16);
implement_get_value_for_built_in_type!(i32, GxB_Scalar_extractElement_INT32);
implement_get_value_for_built_in_type!(i64, GxB_Scalar_extractElement_INT64);
implement_get_value_for_built_in_type!(u8, GxB_Scalar_extractElement_UINT8);
implement_get_value_for_built_in_type!(u16, GxB_Scalar_extractElement_UINT16);
implement_get_value_for_built_in_type!(u32, GxB_Scalar_extractElement_UINT32);
implement_get_value_for_built_in_type!(u64, GxB_Scalar_extractElement_UINT64);
implement_get_value_for_built_in_type!(f32, GxB_Scalar_extractElement_FP32);
implement_get_value_for_built_in_type!(f64, GxB_Scalar_extractElement_FP64);

#[cfg(test)]
mod tests {

    // #[macro_use(implement_value_type_for_custom_type)]

    use super::*;
    use crate::context::Mode;

    // use crate::value_type::{GraphblasFloat32, GraphblasInt32};

    #[test]
    fn new_scalar() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let sparse_scalar = SparseScalar::<i32>::new(&context).unwrap();

        assert_eq!(0, sparse_scalar.number_of_stored_elements().unwrap());
    }

    #[test]
    fn clone_scalar() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let sparse_scalar = SparseScalar::<f32>::new(&context).unwrap();

        let clone_of_sparse_scalar = sparse_scalar.clone();

        // TODO: implement and test equality operator
        assert_eq!(
            0,
            clone_of_sparse_scalar.number_of_stored_elements().unwrap()
        );
    }

    #[test]
    fn test_set_value() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let mut sparse_scalar = SparseScalar::<i32>::new(&context).unwrap();

        sparse_scalar.set_value(&2).unwrap();

        assert_eq!(1, sparse_scalar.number_of_stored_elements().unwrap());
    }

    #[test]
    fn clear_value_from_scalar() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let mut sparse_scalar = SparseScalar::<i32>::new(&context).unwrap();

        sparse_scalar.set_value(&2).unwrap();

        assert_eq!(1, sparse_scalar.number_of_stored_elements().unwrap());

        assert_eq!(2, sparse_scalar.get_value().unwrap());

        sparse_scalar.clear().unwrap();

        assert_eq!(sparse_scalar.number_of_stored_elements().unwrap(), 0)
    }

    #[test]
    fn test_get_value() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let mut sparse_scalar = SparseScalar::<i32>::new(&context).unwrap();

        sparse_scalar.set_value(&2).unwrap();

        assert_eq!(1, sparse_scalar.number_of_stored_elements().unwrap());

        assert_eq!(2, sparse_scalar.get_value().unwrap());
    }
}
