mod coordinate;
mod element;
mod size;
mod sparse_matrix;
// mod constant_sized_sparse_matrix;

pub use coordinate::Coordinate;
pub use element::{MatrixElement, MatrixElementList};
pub use size::Size;
pub use sparse_matrix::{
    FromMatrixElementList, GetMatrixElement, GetMatrixElementList, GetMatrixElementValue,
    SetMatrixElement, SparseMatrix,
};
