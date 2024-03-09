use suitesparse_graphblas_sys::GrB_Matrix_removeElement;

use crate::{
    collections::sparse_matrix::{GetCoordinateIndices, GetGraphblasSparseMatrix, SparseMatrix},
    context::CallGraphBlasContext,
    error::SparseLinearAlgebraError,
    index::{ElementIndex, IndexConversion},
    value_type::ValueType,
};

pub trait DropSparseMatrixElement {
    fn drop_element_with_coordinate(
        &mut self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<(), SparseLinearAlgebraError>;
    fn drop_element(
        &mut self,
        row_index: &ElementIndex,
        column_index: &ElementIndex,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> DropSparseMatrixElement for SparseMatrix<T> {
    fn drop_element_with_coordinate(
        &mut self,
        coordinate: &impl GetCoordinateIndices,
    ) -> Result<(), SparseLinearAlgebraError> {
        drop_sparse_matrix_element_with_coordinate(self, coordinate)
    }
    fn drop_element(
        &mut self,
        row_index: &ElementIndex,
        column_index: &ElementIndex,
    ) -> Result<(), SparseLinearAlgebraError> {
        drop_sparse_matrix_element(self, row_index, column_index)
    }
}

pub fn drop_sparse_matrix_element(
    matrix: &mut impl GetGraphblasSparseMatrix,
    row_index: &ElementIndex,
    column_index: &ElementIndex,
) -> Result<(), SparseLinearAlgebraError> {
    let row_index_to_delete = row_index.as_graphblas_index()?;
    let column_index_to_delete = column_index.as_graphblas_index()?;

    matrix.context_ref().call(
        || unsafe {
            GrB_Matrix_removeElement(
                matrix.graphblas_matrix(),
                row_index_to_delete,
                column_index_to_delete,
            )
        },
        unsafe { &matrix.graphblas_matrix() },
    )?;
    Ok(())
}

pub fn drop_sparse_matrix_element_with_coordinate(
    matrix: &mut impl GetGraphblasSparseMatrix,
    coordinate: &impl GetCoordinateIndices,
) -> Result<(), SparseLinearAlgebraError> {
    let row_index_to_delete = coordinate.row_index().to_graphblas_index()?;
    let column_index_to_delete = coordinate.column_index().to_graphblas_index()?;

    matrix.context_ref().call(
        || unsafe {
            GrB_Matrix_removeElement(
                matrix.graphblas_matrix(),
                row_index_to_delete,
                column_index_to_delete,
            )
        },
        unsafe { &matrix.graphblas_matrix() },
    )?;
    Ok(())
}
