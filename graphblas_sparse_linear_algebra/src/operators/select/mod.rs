// mod diagonal_index;
mod matrix;
mod vector;

// pub use diagonal_index::DiagonalIndex;

pub use matrix::{
    MatrixSelector, SelectMatrixEqualToScalar, SelectMatrixGreaterThanOrEqualToScalar,
    SelectMatrixGreaterThanScalar, SelectMatrixLessThanOrEqualToScalar, SelectMatrixLessThanScalar,
    SelectMatrixNotEqualToScalar,
};
pub use vector::{
    SelectVectorEqualToScalar, SelectVectorGreaterThanOrEqualToScalar,
    SelectVectorGreaterThanScalar, SelectVectorLessThanOrEqualToScalar, SelectVectorLessThanScalar,
    SelectVectorNotEqualToScalar, VectorSelector,
};
