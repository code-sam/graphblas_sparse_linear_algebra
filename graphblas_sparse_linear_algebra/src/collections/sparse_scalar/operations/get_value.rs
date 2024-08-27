use std::mem::MaybeUninit;

use crate::collections::sparse_scalar::GetGraphblasSparseScalar;
use crate::collections::sparse_scalar::SparseScalar;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::GraphblasErrorType;
use crate::error::LogicErrorType;
use crate::error::SparseLinearAlgebraError;
use crate::error::SparseLinearAlgebraErrorType;
use crate::graphblas_bindings::{
    GrB_Scalar_extractElement_BOOL, GrB_Scalar_extractElement_FP32, GrB_Scalar_extractElement_FP64,
    GrB_Scalar_extractElement_INT16, GrB_Scalar_extractElement_INT32,
    GrB_Scalar_extractElement_INT64, GrB_Scalar_extractElement_INT8,
    GrB_Scalar_extractElement_UINT16, GrB_Scalar_extractElement_UINT32,
    GrB_Scalar_extractElement_UINT64, GrB_Scalar_extractElement_UINT8,
};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::ConvertScalar;
use crate::value_type::ValueType;

pub trait GetScalarValue<T: ValueType + Default> {
    fn value(&self) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn value_or_default(&self) -> Result<T, SparseLinearAlgebraError>;
}

impl<T: ValueType + Default + GetScalarValueTyped<T>> GetScalarValue<T> for SparseScalar<T> {
    fn value(&self) -> Result<Option<T>, SparseLinearAlgebraError> {
        T::value(self)
    }

    fn value_or_default(&self) -> Result<T, SparseLinearAlgebraError> {
        T::value_or_default(self)
    }
}

pub trait GetScalarValueTyped<T: ValueType + Default> {
    fn value(scalar: &SparseScalar<T>) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn value_or_default(scalar: &SparseScalar<T>) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_scalar_value_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetScalarValueTyped<$value_type> for $value_type {
            fn value(
                scalar: &SparseScalar<$value_type>,
            ) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                let mut value: MaybeUninit<$graphblas_implementation_type> = MaybeUninit::uninit();

                let result = scalar.context_ref().call(
                    || unsafe {
                        $get_element_function(value.as_mut_ptr(), scalar.graphblas_scalar())
                    },
                    unsafe { &scalar.graphblas_scalar() },
                );

                match result {
                    Ok(_) => {
                        let value = unsafe { value.assume_init() };
                        Ok(Some(value.to_type()?))
                    }
                    Err(error) => match error.error_type() {
                        SparseLinearAlgebraErrorType::LogicErrorType(
                            LogicErrorType::GraphBlas(GraphblasErrorType::NoValue),
                        ) => Ok(None),
                        _ => Err(error),
                    },
                }
            }

            fn value_or_default(
                scalar: &SparseScalar<$value_type>,
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                Ok(<$value_type>::value(scalar)?.unwrap_or_default())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_scalar_value_for_built_in_type,
    GrB_Scalar_extractElement
);
