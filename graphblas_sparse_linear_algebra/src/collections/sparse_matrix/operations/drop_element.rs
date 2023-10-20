use suitesparse_graphblas_sys::GrB_Matrix_removeElement;

use crate::{
    collections::sparse_matrix::{Coordinate, GetGraphblasSparseMatrix, SparseMatrix},
    context::{CallGraphBlasContext, GetContext},
    error::SparseLinearAlgebraError,
    index::IndexConversion,
    value_type::ValueType,
};

pub trait DropSparseMatrixElement {
    fn drop_element(&mut self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> DropSparseMatrixElement for SparseMatrix<T> {
    fn drop_element(&mut self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError> {
        drop_sparse_matrix_element(self, coordinate)
    }
}

pub fn drop_sparse_matrix_element(
    matrix: &mut (impl GetGraphblasSparseMatrix + GetContext),
    coordinate: Coordinate,
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
