use suitesparse_graphblas_sys::GrB_Matrix_resize;

use crate::{
    collections::sparse_matrix::{
        size::GetMatrixDimensions, GetGraphblasSparseMatrix, Size, SparseMatrix,
    },
    context::{CallGraphBlasContext, GetContext},
    error::SparseLinearAlgebraError,
    index::IndexConversion,
    value_type::ValueType,
};

pub trait ResizeSparseMatrix {
    /// All elements of self with an index coordinate outside of the new size are dropped.
    fn resize(&mut self, new_size: &Size) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> ResizeSparseMatrix for SparseMatrix<T> {
    /// All elements of self with an index coordinate outside of the new size are dropped.
    fn resize(&mut self, new_size: &Size) -> Result<(), SparseLinearAlgebraError> {
        resize_sparse_matrix(self, new_size)
    }
}

/// All elements of self with an index coordinate outside of the new size are dropped.
pub fn resize_sparse_matrix(
    matrix: &mut impl GetGraphblasSparseMatrix,
    new_size: &Size,
) -> Result<(), SparseLinearAlgebraError> {
    let new_row_height = new_size.row_height_ref().to_graphblas_index()?;
    let new_column_width = new_size.column_width_ref().to_graphblas_index()?;

    matrix.context_ref().call(
        || unsafe {
            GrB_Matrix_resize(matrix.graphblas_matrix(), new_row_height, new_column_width)
        },
        unsafe { &matrix.graphblas_matrix() },
    )?;
    Ok(())
}
