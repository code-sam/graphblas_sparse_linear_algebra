use crate::collections::sparse_matrix::SparseMatrix;
use crate::{
    collections::sparse_matrix::{Coordinate, MatrixElement},
    error::SparseLinearAlgebraError,
    value_type::ValueType,
};

use super::{GetGraphblasMatrixElementValue, GetMatrixElementValue};

pub trait GetMatrixElement<T: ValueType> {
    fn get_element(
        &self,
        coordinate: Coordinate,
    ) -> Result<Option<MatrixElement<T>>, SparseLinearAlgebraError>;

    fn get_element_or_default(
        &self,
        coordinate: Coordinate,
    ) -> Result<MatrixElement<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType + Default + GetGraphblasMatrixElementValue<T>> GetMatrixElement<T>
    for SparseMatrix<T>
{
    fn get_element(
        &self,
        coordinate: Coordinate,
    ) -> Result<Option<MatrixElement<T>>, SparseLinearAlgebraError> {
        match self.get_element_value(&coordinate)? {
            Some(value) => Ok(Some(MatrixElement::new(coordinate, value))),
            None => Ok(None),
        }
    }

    fn get_element_or_default(
        &self,
        coordinate: Coordinate,
    ) -> Result<MatrixElement<T>, SparseLinearAlgebraError> {
        let value = self.get_element_value_or_default(&coordinate)?;
        Ok(MatrixElement::new(coordinate, value))
    }
}
