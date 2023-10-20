use std::mem::MaybeUninit;

use suitesparse_graphblas_sys::{GrB_Index, GrB_Matrix_ncols, GrB_Matrix_nrows};

use crate::index::IndexConversion;
use crate::{
    collections::sparse_matrix::{GetGraphblasSparseMatrix, Size, SparseMatrix},
    context::{CallGraphBlasContext, GetContext},
    error::SparseLinearAlgebraError,
    index::ElementIndex,
    value_type::ValueType,
};

pub trait GetSparseMatrixSize {
    fn column_width(&self) -> Result<ElementIndex, SparseLinearAlgebraError>;
    fn row_height(&self) -> Result<ElementIndex, SparseLinearAlgebraError>;
    fn size(&self) -> Result<Size, SparseLinearAlgebraError>;
}

impl<T: ValueType> GetSparseMatrixSize for SparseMatrix<T> {
    fn column_width(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        sparse_matrix_column_width(self)
    }

    fn row_height(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        sparse_matrix_row_height(self)
    }

    fn size(&self) -> Result<Size, SparseLinearAlgebraError> {
        sparse_matrix_size(self)
    }
}

pub fn sparse_matrix_column_width(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
) -> Result<ElementIndex, SparseLinearAlgebraError> {
    let mut column_width: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
    matrix.context_ref().call(
        || unsafe { GrB_Matrix_ncols(column_width.as_mut_ptr(), matrix.graphblas_matrix()) },
        unsafe { &matrix.graphblas_matrix() },
    )?;
    let column_width = unsafe { column_width.assume_init() };
    Ok(ElementIndex::from_graphblas_index(column_width)?)
}

pub fn sparse_matrix_row_height(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
) -> Result<ElementIndex, SparseLinearAlgebraError> {
    let mut row_height: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
    matrix.context_ref().call(
        || unsafe { GrB_Matrix_nrows(row_height.as_mut_ptr(), matrix.graphblas_matrix()) },
        unsafe { &matrix.graphblas_matrix() },
    )?;
    let row_height = unsafe { row_height.assume_init() };
    Ok(ElementIndex::from_graphblas_index(row_height)?)
}

pub fn sparse_matrix_size(
    matrix: &(impl GetGraphblasSparseMatrix + GetContext),
) -> Result<Size, SparseLinearAlgebraError> {
    Ok(Size::new(
        sparse_matrix_row_height(matrix)?,
        sparse_matrix_column_width(matrix)?,
    ))
}
