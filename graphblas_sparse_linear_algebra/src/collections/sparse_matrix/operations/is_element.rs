use suitesparse_graphblas_sys::GxB_Matrix_isStoredElement;

use crate::{
    collections::sparse_matrix::{Coordinate, GetGraphblasSparseMatrix, SparseMatrix},
    context::{CallGraphBlasContext, GetContext},
    error::{
        GraphblasErrorType, LogicErrorType, SparseLinearAlgebraError, SparseLinearAlgebraErrorType,
    },
    index::IndexConversion,
    value_type::ValueType,
};

pub trait IsSparseMatrixElement {
    fn is_element(&self, coordinate: Coordinate) -> Result<bool, SparseLinearAlgebraError>;
    fn try_is_element(&self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> IsSparseMatrixElement for SparseMatrix<T> {
    fn is_element(&self, coordinate: Coordinate) -> Result<bool, SparseLinearAlgebraError> {
        is_sparse_matrix_element(self, coordinate)
    }

    fn try_is_element(&self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError> {
        try_is_sparse_matrix_element(self, coordinate)
    }
}

pub fn is_sparse_matrix_element(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
    coordinate: Coordinate,
) -> Result<bool, SparseLinearAlgebraError> {
    let row_index = coordinate.row_index().to_graphblas_index()?;
    let column_index = coordinate.column_index().to_graphblas_index()?;

    let result = matrix.context_ref().call(
        || unsafe {
            GxB_Matrix_isStoredElement(matrix.graphblas_matrix(), row_index, column_index)
        },
        unsafe { &matrix.graphblas_matrix() },
    );
    match result {
        Ok(_) => Ok(true),
        Err(error) => match error.error_type() {
            SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                GraphblasErrorType::NoValue,
            )) => Ok(false),
            _ => Err(error),
        },
    }
}

pub fn try_is_sparse_matrix_element(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
    coordinate: Coordinate,
) -> Result<(), SparseLinearAlgebraError> {
    let row_index = coordinate.row_index().to_graphblas_index()?;
    let column_index = coordinate.column_index().to_graphblas_index()?;

    let result = matrix.context_ref().call(
        || unsafe {
            GxB_Matrix_isStoredElement(matrix.graphblas_matrix(), row_index, column_index)
        },
        unsafe { &matrix.graphblas_matrix() },
    );
    match result {
        Ok(_) => Ok(()),
        Err(error) => Err(error),
    }
}
