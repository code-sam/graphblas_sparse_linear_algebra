use crate::collections::sparse_scalar::GetGraphblasSparseScalar;
use crate::collections::sparse_scalar::SparseScalar;
use crate::context::CallGraphBlasContext;
use crate::value_type::ConvertScalar;
use crate::{
    error::SparseLinearAlgebraError,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};

use crate::graphblas_bindings::{
    GrB_Scalar_setElement_BOOL, GrB_Scalar_setElement_FP32, GrB_Scalar_setElement_FP64,
    GrB_Scalar_setElement_INT16, GrB_Scalar_setElement_INT32, GrB_Scalar_setElement_INT64,
    GrB_Scalar_setElement_INT8, GrB_Scalar_setElement_UINT16, GrB_Scalar_setElement_UINT32,
    GrB_Scalar_setElement_UINT64, GrB_Scalar_setElement_UINT8,
};

pub trait SetScalarValue<T: ValueType> {
    fn set_value(&mut self, value: T) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType + SetScalarValueTyped<T>> SetScalarValue<T> for SparseScalar<T> {
    fn set_value(&mut self, value: T) -> Result<(), SparseLinearAlgebraError> {
        T::set_graphblas_scalar_value(self, value)
    }
}

pub trait SetScalarValueTyped<T: ValueType> {
    fn set_graphblas_scalar_value(
        scalar: &mut impl GetGraphblasSparseScalar,
        value: T,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_set_value_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ident, $add_element_function:ident) => {
        impl SetScalarValueTyped<$value_type> for $value_type {
            fn set_graphblas_scalar_value(
                scalar: &mut impl GetGraphblasSparseScalar,
                value: $value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let element_value = value.to_type()?;
                scalar.context_ref().call(
                    || unsafe { $add_element_function(scalar.graphblas_scalar(), element_value) },
                    unsafe { scalar.graphblas_scalar_ref() },
                )?;
                Ok(())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_set_value_for_built_in_type,
    GrB_Scalar_setElement
);
