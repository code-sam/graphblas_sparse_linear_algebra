use std::mem::MaybeUninit;

use crate::collections::sparse_matrix::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_matrix::SparseMatrix;
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::error::GraphblasErrorType;
use crate::error::LogicErrorType;
use crate::error::SparseLinearAlgebraErrorType;
use crate::index::IndexConversion;
use crate::{
    collections::sparse_matrix::Coordinate,
    error::SparseLinearAlgebraError,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types_and_graphblas_function,
        ValueType,
    },
};

use crate::graphblas_bindings::{
    GrB_Matrix_extractElement_BOOL, GrB_Matrix_extractElement_FP32, GrB_Matrix_extractElement_FP64,
    GrB_Matrix_extractElement_INT16, GrB_Matrix_extractElement_INT32,
    GrB_Matrix_extractElement_INT64, GrB_Matrix_extractElement_INT8,
    GrB_Matrix_extractElement_UINT16, GrB_Matrix_extractElement_UINT32,
    GrB_Matrix_extractElement_UINT64, GrB_Matrix_extractElement_UINT8,
};

pub trait GetSparseMatrixElementValue<T: ValueType> {
    fn get_element_value(
        &self,
        coordinate: &Coordinate,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn get_element_value_or_default(
        &self,
        coordinate: &Coordinate,
    ) -> Result<T, SparseLinearAlgebraError>;
}

impl<T: ValueType + Default + GetSparseMatrixElementValueTyped<T>> GetSparseMatrixElementValue<T>
    for SparseMatrix<T>
{
    fn get_element_value(
        &self,
        coordinate: &Coordinate,
    ) -> Result<Option<T>, SparseLinearAlgebraError> {
        T::get_element_value(self, coordinate)
    }

    fn get_element_value_or_default(
        &self,
        coordinate: &Coordinate,
    ) -> Result<T, SparseLinearAlgebraError> {
        T::get_element_value_or_default(self, coordinate)
    }
}

pub trait GetSparseMatrixElementValueTyped<T: ValueType + Default> {
    fn get_element_value(
        matrix: &SparseMatrix<T>,
        coordinate: &Coordinate,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn get_element_value_or_default(
        matrix: &SparseMatrix<T>,
        coordinate: &Coordinate,
    ) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_value {
    ($value_type:ty, $get_element_function:ident) => {
        impl GetSparseMatrixElementValueTyped<$value_type> for $value_type {
            fn get_element_value(
                matrix: &SparseMatrix<$value_type>,
                coordinate: &Coordinate,
            ) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                // let mut value: MaybeUninit<$value_type> = MaybeUninit::uninit();
                let mut value = MaybeUninit::uninit();
                let row_index_to_get = coordinate.row_index().to_graphblas_index()?;
                let column_index_to_get = coordinate.column_index().to_graphblas_index()?;

                let result = matrix.context_ref().call(
                    || unsafe {
                        $get_element_function(
                            value.as_mut_ptr(),
                            matrix.graphblas_matrix(),
                            row_index_to_get,
                            column_index_to_get,
                        )
                    },
                    unsafe { &matrix.graphblas_matrix() },
                );

                match result {
                    Ok(_) => {
                        let value = unsafe { value.assume_init() };
                        // Casting to support isize and usize, redundant for other types. TODO: review performance improvements
                        Ok(Some(value.try_into().unwrap()))
                    }
                    Err(error) => match error.error_type() {
                        SparseLinearAlgebraErrorType::LogicErrorType(
                            LogicErrorType::GraphBlas(GraphblasErrorType::NoValue),
                        ) => Ok(None),
                        _ => Err(error),
                    },
                }
            }

            fn get_element_value_or_default(
                matrix: &SparseMatrix<$value_type>,
                coordinate: &Coordinate,
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                match <$value_type>::get_element_value(matrix, coordinate)? {
                    Some(value) => Ok(value),
                    None => Ok(<$value_type>::default()),
                }
            }
        }
    };
}

implement_macro_for_all_value_types_and_graphblas_function!(
    implement_get_element_value,
    GrB_Matrix_extractElement
);
