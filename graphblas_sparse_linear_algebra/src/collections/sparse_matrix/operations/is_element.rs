use suitesparse_graphblas_sys::GxB_Matrix_isStoredElement;

use crate::{
    collections::sparse_matrix::{GetCoordinateIndices, GetGraphblasSparseMatrix, SparseMatrix},
    context::{CallGraphBlasContext, GetContext},
    error::{
        GraphblasErrorType, LogicErrorType, SparseLinearAlgebraError, SparseLinearAlgebraErrorType,
    },
    index::{ElementIndex, IndexConversion},
    value_type::ValueType,
};

pub trait IsSparseMatrixElement {
    fn is_element(
        &self,
        row_index: &ElementIndex,
        column_index: &ElementIndex,
    ) -> Result<bool, SparseLinearAlgebraError>;
    fn try_is_element(
        &self,
        row_index: &ElementIndex,
        column_index: &ElementIndex,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn is_element_at_coordinate(
        &self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<bool, SparseLinearAlgebraError>;
    fn try_is_element_at_coordinate(
        &self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> IsSparseMatrixElement for SparseMatrix<T> {
    fn is_element(
        &self,
        row_index: &ElementIndex,
        column_index: &ElementIndex,
    ) -> Result<bool, SparseLinearAlgebraError> {
        is_sparse_matrix_element(self, row_index, column_index)
    }

    fn try_is_element(
        &self,
        row_index: &ElementIndex,
        column_index: &ElementIndex,
    ) -> Result<(), SparseLinearAlgebraError> {
        try_is_sparse_matrix_element(self, row_index, column_index)
    }

    fn is_element_at_coordinate(
        &self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<bool, SparseLinearAlgebraError> {
        is_sparse_matrix_element_at_coordinate(self, coordinate)
    }

    fn try_is_element_at_coordinate(
        &self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<(), SparseLinearAlgebraError> {
        try_is_sparse_matrix_element_at_coordinate(self, coordinate)
    }
}

pub fn is_sparse_matrix_element(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
    row_index: &ElementIndex,
    column_index: &ElementIndex,
) -> Result<bool, SparseLinearAlgebraError> {
    let row_index = row_index.as_graphblas_index()?;
    let column_index = column_index.as_graphblas_index()?;

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

pub fn is_sparse_matrix_element_at_coordinate(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
    coordinate: &impl GetCoordinateIndices,
) -> Result<bool, SparseLinearAlgebraError> {
    is_sparse_matrix_element(
        matrix,
        coordinate.row_index_ref(),
        coordinate.column_index_ref(),
    )
}

pub fn try_is_sparse_matrix_element(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
    row_index: &ElementIndex,
    column_index: &ElementIndex,
) -> Result<(), SparseLinearAlgebraError> {
    let row_index = row_index.as_graphblas_index()?;
    let column_index = column_index.as_graphblas_index()?;

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

pub fn try_is_sparse_matrix_element_at_coordinate(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
    coordinate: &impl GetCoordinateIndices,
) -> Result<(), SparseLinearAlgebraError> {
    try_is_sparse_matrix_element(
        matrix,
        coordinate.row_index_ref(),
        coordinate.column_index_ref(),
    )
}
