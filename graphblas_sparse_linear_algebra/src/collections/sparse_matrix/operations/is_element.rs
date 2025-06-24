use suitesparse_graphblas_sys::GxB_Matrix_isStoredElement;

use crate::collections::sparse_matrix::{
    ColumnIndex, GetCoordinateIndices, GetGraphblasSparseMatrix, RowIndex, SparseMatrix,
};
use crate::context::CallGraphBlasContext;
use crate::error::{
    GraphblasErrorType, LogicErrorType, SparseLinearAlgebraError, SparseLinearAlgebraErrorType,
};
use crate::index::IndexConversion;
use crate::value_type::ValueType;

pub trait IsSparseMatrixElement {
    fn is_element(
        &self,
        row_index: RowIndex,
        column_index: ColumnIndex,
    ) -> Result<bool, SparseLinearAlgebraError>;
    fn try_is_element(
        &self,
        row_index: RowIndex,
        column_index: ColumnIndex,
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
        row_index: RowIndex,
        column_index: ColumnIndex,
    ) -> Result<bool, SparseLinearAlgebraError> {
        is_sparse_matrix_element(self, row_index, column_index)
    }

    fn try_is_element(
        &self,
        row_index: RowIndex,
        column_index: ColumnIndex,
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
    matrix: &impl GetGraphblasSparseMatrix,
    row_index: RowIndex,
    column_index: ColumnIndex,
) -> Result<bool, SparseLinearAlgebraError> {
    let row_index = row_index.as_graphblas_index()?;
    let column_index = column_index.as_graphblas_index()?;

    let result = matrix.context_ref().call(
        || unsafe {
            GxB_Matrix_isStoredElement(matrix.graphblas_matrix(), row_index, column_index)
        },
        unsafe { &matrix.graphblas_matrix() },
    );
    println!("{:?}", result);
    match result {
        Ok(_) => Ok(true),
        Err(error) => match error.error_type() {
            SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                GraphblasErrorType::NoValue,
            )) => Ok(false),
            SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                GraphblasErrorType::IndexOutOfBounds,
            )) => Ok(false),
            SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                GraphblasErrorType::InvalidIndex,
            )) => Ok(false),
            _ => Err(error),
        },
    }
}

pub fn is_sparse_matrix_element_at_coordinate(
    matrix: &impl GetGraphblasSparseMatrix,
    coordinate: &impl GetCoordinateIndices,
) -> Result<bool, SparseLinearAlgebraError> {
    is_sparse_matrix_element(matrix, coordinate.row_index(), coordinate.column_index())
}

pub fn try_is_sparse_matrix_element(
    matrix: &impl GetGraphblasSparseMatrix,
    row_index: RowIndex,
    column_index: ColumnIndex,
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
    matrix: &impl GetGraphblasSparseMatrix,
    coordinate: &impl GetCoordinateIndices,
) -> Result<(), SparseLinearAlgebraError> {
    try_is_sparse_matrix_element(matrix, coordinate.row_index(), coordinate.column_index())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        collections::sparse_matrix::{
            operations::SetSparseMatrixElement, ColumnIndex, Coordinate, MatrixElement, RowIndex,
            Size, SparseMatrix,
        },
        context::Context,
    };

    #[test]
    fn is_element_outside_matrix_size() {
        let context = Context::init_default().unwrap();

        let target_height: RowIndex = 10;
        let target_width: ColumnIndex = 5;
        let size = Size::new(target_height, target_width);

        let mut sparse_matrix = SparseMatrix::<i32>::new(context, size).unwrap();

        sparse_matrix
            .set_element(MatrixElement::from_triple(1, 2, 3))
            .unwrap();
        sparse_matrix
            .set_element(MatrixElement::from_triple(1, 4, 4))
            .unwrap();

        assert!(!sparse_matrix
            .is_element(target_height, target_width)
            .unwrap());
    }
}
