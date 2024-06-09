use std::convert::TryInto;

use crate::error::{SparseLinearAlgebraError, SystemError};
use crate::graphblas_bindings::GrB_Index;

pub type ElementIndex = usize;
pub type ElementCount = ElementIndex;

pub trait IndexConversion {
    fn to_graphblas_index(&self) -> Result<GrB_Index, SparseLinearAlgebraError>;
    fn as_graphblas_index(self) -> Result<GrB_Index, SparseLinearAlgebraError>;
    fn from_graphblas_index(index: GrB_Index) -> Result<ElementIndex, SparseLinearAlgebraError>;
}

// cannot implement From and Into for type defined in another crate. This would need the new type pattern.
// impl From<GrB_Index> for ElementIndex {
//     fn from(self) -> Self {
//         self.from_graphblas_index()
//     }
// }
// impl Into<ElementIndex> for GrB_Index {
//     fn into(self) -> Self {
//         self.as_graphblas_index()
//     }
// }

impl IndexConversion for ElementIndex {
    // TODO: consider to use try_into()
    fn to_graphblas_index(&self) -> Result<GrB_Index, SparseLinearAlgebraError> {
        (*self).as_graphblas_index()
    }

    fn as_graphblas_index(self) -> Result<GrB_Index, SparseLinearAlgebraError> {
        let graphblas_index: GrB_Index;
        match self.try_into() {
            Ok(as_graphblas_index) => graphblas_index = as_graphblas_index,
            Err(error) => return Err(SystemError::from(error).into()),
        }
        Ok(graphblas_index)
    }

    fn from_graphblas_index(index: GrB_Index) -> Result<Self, SparseLinearAlgebraError> {
        let element_index: ElementIndex;
        match index.try_into() {
            Ok(as_element_index) => element_index = as_element_index,
            Err(error) => return Err(SystemError::from(error).into()),
        }
        Ok(element_index)
    }
}
