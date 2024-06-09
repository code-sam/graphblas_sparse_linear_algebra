use std::mem::MaybeUninit;

use crate::collections::sparse_matrix::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_matrix::ColumnIndex;
use crate::collections::sparse_matrix::GetCoordinateIndices;
use crate::collections::sparse_matrix::RowIndex;
use crate::collections::sparse_matrix::SparseMatrix;
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::error::GraphblasErrorType;
use crate::error::LogicErrorType;
use crate::error::SparseLinearAlgebraErrorType;
use crate::index::ElementIndex;
use crate::index::IndexConversion;
use crate::{
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
    fn element_value(
        &self,
        row_index: &RowIndex,
        column_index: &ColumnIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn element_value_or_default(
        &self,
        row_index: &RowIndex,
        column_index: &ColumnIndex,
    ) -> Result<T, SparseLinearAlgebraError>;

    fn element_value_at_coordinate(
        &self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn element_value_or_default_at_coordinate(
        &self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<T, SparseLinearAlgebraError>;
}

impl<T: ValueType + Default + GetSparseMatrixElementValueTyped<T>> GetSparseMatrixElementValue<T>
    for SparseMatrix<T>
{
    fn element_value(
        &self,
        row_index: &RowIndex,
        column_index: &ColumnIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError> {
        T::element_value(self, row_index, column_index)
    }
    fn element_value_or_default(
        &self,
        row_index: &RowIndex,
        column_index: &ColumnIndex,
    ) -> Result<T, SparseLinearAlgebraError> {
        T::element_value_or_default(self, row_index, column_index)
    }

    fn element_value_at_coordinate(
        &self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<Option<T>, SparseLinearAlgebraError> {
        T::element_value_at_coordinate(self, coordinate)
    }

    fn element_value_or_default_at_coordinate(
        &self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<T, SparseLinearAlgebraError> {
        T::element_value_or_default_at_coordinate(self, coordinate)
    }
}

pub trait GetSparseMatrixElementValueTyped<T: ValueType + Default> {
    fn element_value(
        matrix: &SparseMatrix<T>,
        row_index: &RowIndex,
        column_index: &ColumnIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn element_value_or_default(
        matrix: &SparseMatrix<T>,
        row_index: &RowIndex,
        column_index: &ColumnIndex,
    ) -> Result<T, SparseLinearAlgebraError>;

    fn element_value_at_coordinate(
        matrix: &SparseMatrix<T>,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn element_value_or_default_at_coordinate(
        matrix: &SparseMatrix<T>,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_value {
    ($value_type:ty, $get_element_function:ident) => {
        impl GetSparseMatrixElementValueTyped<$value_type> for $value_type {
            fn element_value(
                matrix: &SparseMatrix<$value_type>,
                row_index: &RowIndex,
                column_index: &ColumnIndex,
            ) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                let mut value = MaybeUninit::uninit();
                let row_index_to_get = row_index.as_graphblas_index()?;
                let column_index_to_get = column_index.as_graphblas_index()?;

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

            fn element_value_or_default(
                matrix: &SparseMatrix<$value_type>,
                row_index: &RowIndex,
                column_index: &ColumnIndex,
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                match <$value_type>::element_value(matrix, row_index, column_index)? {
                    Some(value) => Ok(value),
                    None => Ok(<$value_type>::default()),
                }
            }

            fn element_value_at_coordinate(
                matrix: &SparseMatrix<$value_type>,
                coordinate: &impl GetCoordinateIndices,
            ) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                <$value_type>::element_value(
                    matrix,
                    coordinate.row_index_ref(),
                    coordinate.column_index_ref(),
                )
            }

            fn element_value_or_default_at_coordinate(
                matrix: &SparseMatrix<$value_type>,
                coordinate: &impl GetCoordinateIndices,
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                <$value_type>::element_value_or_default(
                    matrix,
                    coordinate.row_index_ref(),
                    coordinate.column_index_ref(),
                )
            }
        }
    };
}

implement_macro_for_all_value_types_and_graphblas_function!(
    implement_get_element_value,
    GrB_Matrix_extractElement
);
