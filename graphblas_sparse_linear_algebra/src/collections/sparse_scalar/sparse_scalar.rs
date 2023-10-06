use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

use suitesparse_graphblas_sys::GrB_Info;

use crate::collections::collection::Collection;
use crate::context::{CallGraphBlasContext, Context, ContextTrait};
use crate::error::{
    GraphblasErrorType, LogicErrorType, SparseLinearAlgebraError, SparseLinearAlgebraErrorType,
};
use crate::graphblas_bindings::{
    GrB_Index, GrB_Scalar, GrB_Scalar_clear, GrB_Scalar_dup, GrB_Scalar_extractElement_BOOL,
    GrB_Scalar_extractElement_FP32, GrB_Scalar_extractElement_FP64,
    GrB_Scalar_extractElement_INT16, GrB_Scalar_extractElement_INT32,
    GrB_Scalar_extractElement_INT64, GrB_Scalar_extractElement_INT8,
    GrB_Scalar_extractElement_UINT16, GrB_Scalar_extractElement_UINT32,
    GrB_Scalar_extractElement_UINT64, GrB_Scalar_extractElement_UINT8, GrB_Scalar_free,
    GrB_Scalar_new, GrB_Scalar_nvals, GrB_Scalar_setElement_BOOL, GrB_Scalar_setElement_FP32,
    GrB_Scalar_setElement_FP64, GrB_Scalar_setElement_INT16, GrB_Scalar_setElement_INT32,
    GrB_Scalar_setElement_INT64, GrB_Scalar_setElement_INT8, GrB_Scalar_setElement_UINT16,
    GrB_Scalar_setElement_UINT32, GrB_Scalar_setElement_UINT64, GrB_Scalar_setElement_UINT8,
};
use crate::index::{ElementIndex, IndexConversion};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
    implement_macro_for_all_value_types,
};
use crate::value_type::ConvertScalar;
use crate::value_type::ValueType;

#[derive(Debug)]
pub struct SparseScalar<T: ValueType> {
    context: Arc<Context>,
    scalar: GrB_Scalar,
    value_type: PhantomData<T>,
}

// Mutable access to GrB_Vector shall occur through a write lock on RwLock<GrB_Matrix>.
// Code review must consider that the correct lock is made via
// SparseMatrix::get_write_lock() and SparseMatrix::get_read_lock().
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<T: ValueType> Send for SparseScalar<T> {}
unsafe impl<T: ValueType> Sync for SparseScalar<T> {}

impl<T: ValueType> SparseScalar<T> {
    pub fn new(context: &Arc<Context>) -> Result<Self, SparseLinearAlgebraError> {
        let mut scalar: MaybeUninit<GrB_Scalar> = MaybeUninit::uninit();
        let context = context.to_owned();

        context.call_without_detailed_error_information(|| unsafe {
            GrB_Scalar_new(scalar.as_mut_ptr(), <T>::to_graphblas_type())
        })?;

        let scalar = unsafe { scalar.assume_init() };
        return Ok(SparseScalar {
            context,
            scalar,
            value_type: PhantomData,
        });
    }

    // pub fn from_value(
    //     context: &Arc<Context>,
    //     value: &T,
    // ) -> Result<Self, SparseLinearAlgebraError> {
    //     let mut sparse_scalar = SparseScalar::new(context)?;
    //     sparse_scalar.set_value(value)?;
    //     Ok(sparse_scalar)
    // }
}

// impl<T: ValueType + BuiltInValueType + SetScalarValue<T>> SparseScalar<T> {
//     pub fn from_scalar(context: &Arc<Context>, value: T) -> Result<Self, SparseLinearAlgebraError> {
//         let mut sparse_scalar = SparseScalar::new(context)?;
//         sparse_scalar.set_value(&value)?;
//         Ok(sparse_scalar)
//     }
// }

macro_rules! sparse_scalar_from_scalar {
    ($value_type: ty) => {
        impl SparseScalar<$value_type> {
            pub fn from_value(
                context: &Arc<Context>,
                value: $value_type,
            ) -> Result<Self, SparseLinearAlgebraError> {
                let mut sparse_scalar = SparseScalar::new(context)?;
                sparse_scalar.set_value(&value)?;
                Ok(sparse_scalar)
            }
        }
    };
}
implement_macro_for_all_value_types!(sparse_scalar_from_scalar);

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

impl<T: ValueType> ContextTrait for SparseScalar<T> {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl<T: ValueType> Collection for SparseScalar<T> {
    fn clear(&mut self) -> Result<(), SparseLinearAlgebraError> {
        self.context
            .call_without_detailed_error_information(|| unsafe { GrB_Scalar_clear(self.scalar) })?;
        Ok(())
    }

    fn number_of_stored_elements(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut number_of_values: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context.call(
            || unsafe { GrB_Scalar_nvals(number_of_values.as_mut_ptr(), self.scalar) },
            &self.scalar,
        )?;
        let number_of_values = unsafe { number_of_values.assume_init() };
        Ok(ElementIndex::from_graphblas_index(number_of_values)?)
    }
}

impl<T: ValueType> Drop for SparseScalar<T> {
    fn drop(&mut self) -> () {
        let _ = self.context.call(
            || unsafe { GrB_Scalar_free(&mut self.scalar.to_owned()) },
            &self.scalar,
        );
    }
}

impl<T: ValueType> Clone for SparseScalar<T> {
    fn clone(&self) -> Self {
        let mut scalar_copy: MaybeUninit<GrB_Scalar> = MaybeUninit::uninit();
        self.context
            .call(
                || unsafe { GrB_Scalar_dup(scalar_copy.as_mut_ptr(), self.scalar) },
                &self.scalar,
            )
            .unwrap();

        SparseScalar {
            context: self.context.to_owned(),
            scalar: unsafe { scalar_copy.assume_init() },
            value_type: PhantomData,
        }
    }
}

pub trait GraphblasSparseScalarTrait {
    unsafe fn graphblas_scalar(&self) -> GrB_Scalar;
    unsafe fn graphblas_scalar_ref(&self) -> &GrB_Scalar;
    unsafe fn graphblas_scalar_mut_ref(&mut self) -> &mut GrB_Scalar;
}

impl<T: ValueType> GraphblasSparseScalarTrait for SparseScalar<T> {
    unsafe fn graphblas_scalar(&self) -> GrB_Scalar {
        self.scalar
    }
    unsafe fn graphblas_scalar_ref(&self) -> &GrB_Scalar {
        &self.scalar
    }
    unsafe fn graphblas_scalar_mut_ref(&mut self) -> &mut GrB_Scalar {
        &mut self.scalar
    }
}

// // TODO improve printing format
// // summary data, column aligning
// impl<T: ValueType + GetScalarValue<T> + Default> std::fmt::Display for SparseScalar<T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         let value: T;
//         match self.get_value() {
//             Err(_error) => return Err(std::fmt::Error),
//             Ok(inner_value) => {
//                 value = inner_value;
//             }
//         }
//         writeln! {f,"Number of stored elements: {:?}", self.number_of_stored_elements()?};
//         writeln! {f,"Value: {:?}", value};
//         writeln!(f, "")
//     }
// }

// TODO: make the implementation generic
// TODO improve printing format
// summary data, column aligning
macro_rules! implement_dispay {
    ($value_type:ty) => {
        impl std::fmt::Display for SparseScalar<$value_type> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let value: Option<$value_type>;
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

// pub trait SetScalarValueGAT {
//     type SparseScalar<T> where T: ValueType;

//     fn set_value(&mut self, value: &T) -> Result<(), SparseLinearAlgebraError>;
// }

// impl SetScalarValueGAT for  {

// }

// impl<T: ValueType> SetScalarValue<T> for SparseScalar<T> {
//     fn set_value(&mut self, value: &T) -> Result<(), SparseLinearAlgebraError> {
//         let value = value.to_owned(); // TODO: review if clone can be removed, and if this improves performance
//         convert_to_target_type!(value, $graphblas_implementation_type);
//         self.context.call(
//             || unsafe { GrB_Scalar_setElement(self.scalar, value) },
//             &self.scalar,
//         )?;
//         Ok(())
//     }
// }

trait GraphBlasSetElementFunction<T: ValueType, U: ValueType> {
    fn graphblas_set_element_function() -> unsafe extern "C" fn(GrB_Scalar, U) -> GrB_Info;
}

macro_rules! implement_graphblas_set_element_function {
    ($value_type: ty, $graphblas_implementation_type: ty, $graphblas_function: ident) => {
        impl GraphBlasSetElementFunction<$value_type, $graphblas_implementation_type>
            for $value_type
        {
            fn graphblas_set_element_function(
            ) -> unsafe extern "C" fn(GrB_Scalar, $graphblas_implementation_type) -> GrB_Info {
                return $graphblas_function;
            }
        }
    };
}
implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_graphblas_set_element_function,
    GrB_Scalar_setElement
);

// impl<T: ValueType + BuiltInValueType + ConvertScalar<T, U>, U: ValueType + BuiltInValueType> SetScalarValue<T> for SparseScalar<T> {
//     fn set_value(&mut self, value: &T) -> Result<(), SparseLinearAlgebraError> {
//         let value: U = value.to_type();
//         self.context.call(
//             || unsafe { <T>::graphblas_set_element_function()(self.scalar, value) },
//             &self.scalar,
//         )?;
//         Ok(())
//     }
// }

macro_rules! implement_set_value_for_built_in_type {
    ($value_type:ty) => {
        impl SetScalarValue<$value_type> for SparseScalar<$value_type> {
            fn set_value(&mut self, value: &$value_type) -> Result<(), SparseLinearAlgebraError> {
                let value = value.to_type()?;
                self.context.call(
                    || unsafe {
                        <$value_type>::graphblas_set_element_function()(self.scalar, value)
                    },
                    &self.scalar,
                )?;
                Ok(())
            }
        }
    };
}

implement_macro_for_all_value_types!(implement_set_value_for_built_in_type);

pub trait GetScalarValue<T: ValueType + Default> {
    fn get_value(&self) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn get_value_or_default(&self) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_value_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ty, $get_value_function:ident) => {
        impl GetScalarValue<$value_type> for SparseScalar<$value_type> {
            fn get_value(&self) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                let mut value: MaybeUninit<$graphblas_implementation_type> = MaybeUninit::uninit();

                let result = self.context.call(
                    || unsafe { $get_value_function(value.as_mut_ptr(), self.scalar) },
                    &self.scalar,
                );

                match result {
                    Ok(_) => {
                        let value = unsafe { value.assume_init() };
                        Ok(Some(<$graphblas_implementation_type>::to_type(value)?))
                    }
                    Err(error) => match error.error_type() {
                        SparseLinearAlgebraErrorType::LogicErrorType(
                            LogicErrorType::GraphBlas(GraphblasErrorType::NoValue),
                        ) => Ok(None),
                        _ => Err(error),
                    },
                }
            }

            fn get_value_or_default(&self) -> Result<$value_type, SparseLinearAlgebraError> {
                match self.get_value()? {
                    Some(value) => Ok(value),
                    None => Ok(<$value_type>::default()),
                }
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_value_for_built_in_type,
    GrB_Scalar_extractElement
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

        let clone_of_sparse_scalar = sparse_scalar.to_owned();

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

        assert_eq!(2, sparse_scalar.get_value_or_default().unwrap());

        sparse_scalar.clear().unwrap();

        assert_eq!(sparse_scalar.number_of_stored_elements().unwrap(), 0)
    }

    #[test]
    fn test_get_value() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let mut sparse_scalar = SparseScalar::<i32>::new(&context).unwrap();

        sparse_scalar.set_value(&2).unwrap();

        assert_eq!(1, sparse_scalar.number_of_stored_elements().unwrap());

        assert_eq!(2, sparse_scalar.get_value_or_default().unwrap());
    }
}
