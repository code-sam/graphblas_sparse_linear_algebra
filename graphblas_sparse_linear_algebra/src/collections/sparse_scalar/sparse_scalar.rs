use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

use suitesparse_graphblas_sys::{GrB_Info, GrB_Type};

use crate::collections::collection::Collection;
use crate::collections::sparse_scalar::operations::SetScalarValue;
use crate::context::{CallGraphBlasContext, Context, GetContext};
use crate::error::{
    GraphblasErrorType, LogicErrorType, SparseLinearAlgebraError, SparseLinearAlgebraErrorType,
};
use crate::graphblas_bindings::{
    GrB_Index, GrB_Scalar, GrB_Scalar_clear, GrB_Scalar_dup, GrB_Scalar_free, GrB_Scalar_new,
    GrB_Scalar_nvals,
};
use crate::index::{ElementCount, ElementIndex, IndexConversion};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
    implement_macro_for_all_value_types,
};
use crate::value_type::ConvertScalar;
use crate::value_type::ValueType;

use crate::collections::sparse_scalar::operations::GetScalarValue;

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

pub unsafe fn new_graphblas_scalar(
    context: &Arc<Context>,
    graphblas_value_type: GrB_Type,
) -> Result<GrB_Scalar, SparseLinearAlgebraError> {
    let mut scalar: MaybeUninit<GrB_Scalar> = MaybeUninit::uninit();

    context.call_without_detailed_error_information(|| unsafe {
        GrB_Scalar_new(scalar.as_mut_ptr(), graphblas_value_type)
    })?;

    let scalar = unsafe { scalar.assume_init() };
    return Ok(scalar);
}

impl<T: ValueType> SparseScalar<T> {
    pub fn new(context: Arc<Context>) -> Result<Self, SparseLinearAlgebraError> {
        let scalar = unsafe { new_graphblas_scalar(&context, T::to_graphblas_type())? };
        return Ok(SparseScalar {
            context,
            scalar,
            value_type: PhantomData,
        });
    }

    pub unsafe fn from_graphblas_scalar(
        context: Arc<Context>,
        scalar: GrB_Scalar,
    ) -> Result<SparseScalar<T>, SparseLinearAlgebraError> {
        Ok(SparseScalar {
            context: context.clone(),
            scalar,
            value_type: PhantomData,
        })
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
                context: Arc<Context>,
                value: $value_type,
            ) -> Result<Self, SparseLinearAlgebraError> {
                let mut sparse_scalar = SparseScalar::new(context)?;
                sparse_scalar.set_value(value)?;
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

impl<T: ValueType> GetContext for SparseScalar<T> {
    fn context(&self) -> Arc<Context> {
        self.context.clone()
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

    fn number_of_stored_elements(&self) -> Result<ElementCount, SparseLinearAlgebraError> {
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
        let _ = self
            .context
            .call_without_detailed_error_information(|| unsafe {
                GrB_Scalar_free(&mut self.scalar)
            });
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
            context: self.context.clone(),
            scalar: unsafe { scalar_copy.assume_init() },
            value_type: PhantomData,
        }
    }
}

pub trait GetGraphblasSparseScalar: GetContext {
    unsafe fn graphblas_scalar(&self) -> GrB_Scalar;
    unsafe fn graphblas_scalar_ref(&self) -> &GrB_Scalar;
    unsafe fn graphblas_scalar_mut_ref(&mut self) -> &mut GrB_Scalar;
}

impl<T: ValueType> GetGraphblasSparseScalar for SparseScalar<T> {
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
                match self.value() {
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

#[cfg(test)]
mod tests {

    // #[macro_use(implement_value_type_for_custom_type)]

    use super::*;

    // use crate::value_type::{GraphblasFloat32, GraphblasInt32};

    #[test]
    fn new_scalar() {
        let context = Context::init_default().unwrap();

        let sparse_scalar = SparseScalar::<i32>::new(context).unwrap();

        assert_eq!(0, sparse_scalar.number_of_stored_elements().unwrap());
    }

    #[test]
    fn clone_scalar() {
        let context = Context::init_default().unwrap();

        let sparse_scalar = SparseScalar::<f32>::new(context).unwrap();

        let clone_of_sparse_scalar = sparse_scalar.clone();

        // TODO: implement and test equality operator
        assert_eq!(
            0,
            clone_of_sparse_scalar.number_of_stored_elements().unwrap()
        );
    }

    #[test]
    fn test_set_value() {
        let context = Context::init_default().unwrap();

        let mut sparse_scalar = SparseScalar::<i32>::new(context).unwrap();

        sparse_scalar.set_value(2).unwrap();

        assert_eq!(1, sparse_scalar.number_of_stored_elements().unwrap());
    }

    #[test]
    fn clear_value_from_scalar() {
        let context = Context::init_default().unwrap();

        let mut sparse_scalar = SparseScalar::<i32>::new(context).unwrap();

        sparse_scalar.set_value(2).unwrap();

        assert_eq!(1, sparse_scalar.number_of_stored_elements().unwrap());

        assert_eq!(2, sparse_scalar.value_or_default().unwrap());

        sparse_scalar.clear().unwrap();

        assert_eq!(sparse_scalar.number_of_stored_elements().unwrap(), 0)
    }

    #[test]
    fn test_get_value() {
        let context = Context::init_default().unwrap();

        let mut sparse_scalar = SparseScalar::<i32>::new(context).unwrap();

        sparse_scalar.set_value(2).unwrap();

        assert_eq!(1, sparse_scalar.number_of_stored_elements().unwrap());

        assert_eq!(2, sparse_scalar.value_or_default().unwrap());
    }
}
