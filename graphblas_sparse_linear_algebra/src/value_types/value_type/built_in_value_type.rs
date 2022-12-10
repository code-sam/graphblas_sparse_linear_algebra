use std::panic::catch_unwind;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use suitesparse_graphblas_sys::{
    GrB_BOOL, GrB_FP32, GrB_FP64, GrB_INT16, GrB_INT32, GrB_INT64, GrB_INT8, GrB_Type, GrB_UINT16,
    GrB_UINT32, GrB_UINT64, GrB_UINT8,
};

use crate::{
    error::{LogicError, SparseLinearAlgebraError, SystemError, SystemErrorType},
    value_types::utilities_to_implement_traits_for_all_value_types::{
        implement_macro_for_all_value_types_and_graphblas_function, implement_type_conversion_macro,
    },
};

pub trait BuiltInValueType {
    fn to_graphblas_type() -> GrB_Type;
}

macro_rules! implement_value_type_for_graphblas_built_in_type {
    ($value_type: ty, $graphblas_type_identifier: ident) => {
        impl BuiltInValueType for $value_type {
            fn to_graphblas_type() -> GrB_Type {
                unsafe { $graphblas_type_identifier }
            }
        }
    };
}

implement_macro_for_all_value_types_and_graphblas_function!(
    implement_value_type_for_graphblas_built_in_type,
    GrB
);

pub(crate) trait ConvertScalar<T: BuiltInValueType, U: BuiltInValueType> {
    fn to_type(self) -> Result<U, SparseLinearAlgebraError>;
}

macro_rules! implement_scalar_conversion {
    ($from_type: ty, $to_type: ty, $implement_to_type: ident) => {
        impl ConvertScalar<$from_type, $to_type> for $from_type {
            $implement_to_type!($from_type, $to_type);
        }
    };
}

macro_rules! scalar_indentity_conversion {
    ($from_type: ty, $to_type: ty) => {
        fn to_type(self) -> Result<$to_type, SparseLinearAlgebraError> {
            Ok(self)
        }
    };
}

macro_rules! scalar_conversion {
    ($from_type: ty, $to_type: ty) => {
        fn to_type(self) -> Result<$to_type, SparseLinearAlgebraError> {
            let as_type: Result<$to_type, std::num::TryFromIntError> = self.try_into();
            match as_type {
                Ok(as_type) => Ok(as_type),
                Err(error) => Err(LogicError::from(error).into()),
            }
        }
    };
}

implement_type_conversion_macro!(
    implement_scalar_conversion,
    scalar_indentity_conversion,
    scalar_conversion
);

pub(crate) trait ConvertVector<T: BuiltInValueType, U: BuiltInValueType> {
    fn to_type(self) -> Result<Vec<U>, SparseLinearAlgebraError>;
}

macro_rules! implement_vector_conversion {
    ($from_type: ty, $to_type: ty, $implement_to_type: ident) => {
        impl ConvertVector<$from_type, $to_type> for Vec<$from_type> {
            $implement_to_type!($from_type, $to_type);
        }
    };
}

macro_rules! vector_indentity_conversion {
    ($from_type: ty, $to_type: ty) => {
        fn to_type(self) -> Result<Vec<$to_type>, SparseLinearAlgebraError> {
            Ok(self)
        }
    };
}

macro_rules! vector_conversion {
    ($from_type: ty, $to_type: ty) => {
        fn to_type(self) -> Result<Vec<$to_type>, SparseLinearAlgebraError> {
            let result = catch_unwind(|| {
                let mut shared_errors = Arc::new(Mutex::new(Vec::new()));
                let as_type: Vec<$to_type> = self
                    .into_par_iter()
                    .map(|x| x.try_into())
                    // TODO: review if this also unlocks
                    .filter_map(|r| r.map_err(|e| shared_errors.lock().unwrap().push(e)).ok())
                    .collect();

                let errors = Arc::try_unwrap(shared_errors)
                    .unwrap()
                    .into_inner()
                    .unwrap();

                if errors.is_empty() {
                    return Ok(as_type);
                } else {
                    return Err(LogicError::from(errors[0]).into());
                }
            });

            match result {
                Ok(result) => result,
                Err(error) => Err(SystemError::new(
                    SystemErrorType::PoisonedData,
                    format!("Something went wrong: {:?}", error),
                    None,
                )
                .into()),
            }
        }
    };
}

implement_type_conversion_macro!(
    implement_vector_conversion,
    vector_indentity_conversion,
    vector_conversion
);

// ($variable: ident, $target_type: ty) => {
//     let $variable: Vec<$target_type> = $variable
//         .to_owned()
//         .into_par_iter()
//         .map(|x| x.try_into().unwrap())
//         .collect();
// };
