use std::convert::TryInto;
use std::sync::Arc;

use crate::context::Context;
use crate::value_type::ConvertScalar;
use crate::{
    collections::sparse_scalar::SparseScalar,
    error::{SparseLinearAlgebraError, SystemError},
};

use super::ElementIndex;

pub type DiagonalIndex = isize;
pub type GraphblasDiagionalIndex = i64; // TODO: check when the GraphBLAS specification formally declares the type

pub trait DiagonalIndexConversion {
    fn to_graphblas_index(&self) -> Result<GraphblasDiagionalIndex, SparseLinearAlgebraError>;
    fn as_graphblas_index(self) -> Result<GraphblasDiagionalIndex, SparseLinearAlgebraError>;
    fn from_graphblas_index(
        index: GraphblasDiagionalIndex,
    ) -> Result<DiagonalIndex, SparseLinearAlgebraError>;
    fn to_graphblas_element_index(&self) -> Result<ElementIndex, SparseLinearAlgebraError>;
    fn to_sparse_scalar(
        &self,
        context: Arc<Context>,
    ) -> Result<SparseScalar<GraphblasDiagionalIndex>, SparseLinearAlgebraError>;
}

impl DiagonalIndexConversion for DiagonalIndex {
    fn to_graphblas_index(&self) -> Result<GraphblasDiagionalIndex, SparseLinearAlgebraError> {
        (*self).as_graphblas_index()
    }

    fn as_graphblas_index(self) -> Result<GraphblasDiagionalIndex, SparseLinearAlgebraError> {
        let graphblas_index: i64;
        match self.try_into() {
            Ok(as_graphblas_index) => graphblas_index = as_graphblas_index,
            Err(error) => return Err(SystemError::from(error).into()),
        }
        Ok(graphblas_index)
    }

    fn from_graphblas_index(
        index: GraphblasDiagionalIndex,
    ) -> Result<Self, SparseLinearAlgebraError> {
        let diagonal_index: DiagonalIndex;
        match index.try_into() {
            Ok(as_index) => diagonal_index = as_index,
            Err(error) => return Err(SystemError::from(error).into()),
        }
        Ok(diagonal_index)
    }

    fn to_graphblas_element_index(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let element_index: ElementIndex;
        match (*self).try_into() {
            Ok(as_element_index) => element_index = as_element_index,
            Err(error) => return Err(SystemError::from(error).into()),
        }
        Ok(element_index)
    }

    fn to_sparse_scalar(
        &self,
        context: Arc<Context>,
    ) -> Result<SparseScalar<GraphblasDiagionalIndex>, SparseLinearAlgebraError> {
        Ok(SparseScalar::<GraphblasDiagionalIndex>::from_value(
            context,
            self.to_type()?,
        )?)
    }
}
