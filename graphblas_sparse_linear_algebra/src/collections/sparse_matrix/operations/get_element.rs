use crate::collections::sparse_matrix::SparseMatrix;
use crate::{
    collections::sparse_matrix::{Coordinate, MatrixElement},
    error::SparseLinearAlgebraError,
    value_type::ValueType,
};

use super::{GetSparseMatrixElementValue, GetSparseMatrixElementValueTyped};

pub trait GetSparseMatrixElement<T: ValueType> {
    fn element(
        &self,
        coordinate: Coordinate,
    ) -> Result<Option<MatrixElement<T>>, SparseLinearAlgebraError>;

    fn element_or_default(
        &self,
        coordinate: Coordinate,
    ) -> Result<MatrixElement<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType + Default + GetSparseMatrixElementValueTyped<T>> GetSparseMatrixElement<T>
    for SparseMatrix<T>
{
    fn element(
        &self,
        coordinate: Coordinate,
    ) -> Result<Option<MatrixElement<T>>, SparseLinearAlgebraError> {
        match self.element_value_at_coordinate(&coordinate)? {
            Some(value) => Ok(Some(MatrixElement::new(coordinate, value))),
            None => Ok(None),
        }
    }

    fn element_or_default(
        &self,
        coordinate: Coordinate,
    ) -> Result<MatrixElement<T>, SparseLinearAlgebraError> {
        let value = self.element_value_or_default_at_coordinate(&coordinate)?;
        Ok(MatrixElement::new(coordinate, value))
    }
}
