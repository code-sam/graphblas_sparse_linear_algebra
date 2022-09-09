use std::convert::TryInto;
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
use crate::value_types::utilities_to_implement_traits_for_all_value_types::{
    convert_scalar_to_type, identity_conversion, implement_macro_for_all_value_types,
    implement_macro_for_all_value_types_and_graphblas_function_with_scalar_type_conversion,
    implement_trait_for_all_value_types,
};
use crate::value_types::value_type::{BuiltInValueType, ValueType};

#[derive(Debug)]
pub struct SparseScalar<T: ValueType> {
    context: Arc<Context>,
    scalar: GxB_Scalar,
    value_type: PhantomData<T>,
}

// Mutable access to GrB_Vector shall occur through a write lock on RwLock<GrB_Matrix>.
// Code review must consider that the correct lock is made via
// SparseMatrix::get_write_lock() and SparseMatrix::get_read_lock().
// https://doc.rust-lang.org/nomicon/send-and-sync.html
implement_trait_for_all_value_types!(Send, SparseScalar);
implement_trait_for_all_value_types!(Sync, SparseScalar);

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

// impl<T: ValueType + CustomValueType> SparseScalar<T> {
//     pub fn new_custom_type(
//         value_type: Arc<RegisteredCustomValueType<T>>,
//     ) -> Result<Self, SparseLinearAlgebraError> {
//         let mut scalar: MaybeUninit<GxB_Scalar> = MaybeUninit::uninit();
//         let context = value_type.context();

//         context.call(|| unsafe {
//             GxB_Scalar_new(scalar.as_mut_ptr(), value_type.to_graphblas_type())
//         })?;

//         let scalar = unsafe { scalar.assume_init() };
//         return Ok(SparseScalar {
//             context,
//             scalar,
//             value_type: PhantomData,
//         });
//     }
// }

impl<T: ValueType> SparseScalar<T> {
    pub fn context(&self) -> Arc<Context> {
        self.context.clone()
    }
    pub fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }

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

implement_macro_for_all_value_types!(implement_dispay);

pub trait SetScalarValue<T: ValueType> {
    fn set_value(&mut self, value: &T) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_set_value_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ty, $add_element_function:ident, $convert_to_target_type:ident) => {
        impl SetScalarValue<$value_type> for SparseScalar<$value_type> {
            fn set_value(&mut self, value: &$value_type) -> Result<(), SparseLinearAlgebraError> {
                let value = value.clone(); // TODO: review if clone can be removed, and if this improves performance
                $convert_to_target_type!(value, $graphblas_implementation_type);
                self.context
                    .call(|| unsafe { $add_element_function(self.scalar, value) })?;
                Ok(())
            }
        }
    };
}

implement_macro_for_all_value_types_and_graphblas_function_with_scalar_type_conversion!(
    implement_set_value_for_built_in_type,
    GxB_Scalar_setElement
);

pub trait GetScalarValue<T: ValueType + Default> {
    fn get_value(&self) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_value_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ty, $get_value_function:ident, $convert_to_target_type:ident) => {
        impl GetScalarValue<$value_type> for SparseScalar<$value_type> {
            fn get_value(&self) -> Result<$value_type, SparseLinearAlgebraError> {
                let mut value: MaybeUninit<$graphblas_implementation_type> = MaybeUninit::uninit();

                let result = self
                    .context
                    .call(|| unsafe { $get_value_function(value.as_mut_ptr(), self.scalar) });

                match result {
                    Ok(_) => {
                        let value = unsafe { value.assume_init() };
                        $convert_to_target_type!(value, $value_type);
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

implement_macro_for_all_value_types_and_graphblas_function_with_scalar_type_conversion!(
    implement_get_value_for_built_in_type,
    GxB_Scalar_extractElement
);

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
