use std::sync::Arc;

use crate::context::Context;
use crate::error::SparseLinearAlgebraError;

use crate::value_types::sparse_scalar::{SetScalarValue, SparseScalar};

type GraphblasDiagonalDiagonalIndex = i64;

/// Default selects the main diagonal
pub enum DiagonalIndex {
    Default(),
    Index(GraphblasDiagonalDiagonalIndex),
}

pub(crate) enum DiagonalIndexGraphblasType {
    Default,
    Index(SparseScalar<GraphblasDiagonalDiagonalIndex>),
}

impl DiagonalIndex {
    pub(crate) fn to_graphblas_type(
        &self,
        context: &Arc<Context>,
    ) -> Result<DiagonalIndexGraphblasType, SparseLinearAlgebraError> {
        match self {
            DiagonalIndex::Index(index) => {
                let mut scalar_index =
                    SparseScalar::<GraphblasDiagonalDiagonalIndex>::new(context)?;
                scalar_index.set_value(index)?;

                Ok(DiagonalIndexGraphblasType::Index(scalar_index))
            }
            DiagonalIndex::Default() => Ok(DiagonalIndexGraphblasType::Default),
        }
    }
}

impl From<i64> for DiagonalIndex {
    fn from(index: i64) -> Self {
        DiagonalIndex::Index(index)
    }
}
