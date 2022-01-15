use crate::bindings_to_graphblas_implementation::GrB_Index;
use crate::error::{SparseLinearAlgebraError, SystemError, SystemErrorType};

pub type ElementIndex = usize;

pub trait IndexConversion {
    fn to_graphblas_index(&self) -> Result<GrB_Index, SparseLinearAlgebraError>;
    fn as_graphblas_index(self) -> Result<GrB_Index, SparseLinearAlgebraError>;
    fn from_graphblas_index(index: GrB_Index) -> Result<ElementIndex, SparseLinearAlgebraError>;
    // fn to_usize(&self) -> Result<usize, SystemError>;
}

impl IndexConversion for ElementIndex {
    fn to_graphblas_index(&self) -> Result<GrB_Index, SparseLinearAlgebraError> {
        if *self < (GrB_Index::MAX as Self) {
            // as u64 will truncate to u64::MAX if usize::MAX would be larger than u64::MAX
            Ok(self.clone() as GrB_Index)
        } else {
            Err(SystemError::new(
                SystemErrorType::IndexOutOfBounds,
                String::from("ElementIndex not smaller than GrB_Index::MAX"),
                None,
            )
            .into())
        }
    }

    fn as_graphblas_index(self) -> Result<GrB_Index, SparseLinearAlgebraError> {
        if self < (GrB_Index::MAX as Self) {
            // as u64 will truncate to u64::MAX if usize::MAX would be larger than u64::MAX
            Ok(self as GrB_Index)
        } else {
            Err(SystemError::new(
                SystemErrorType::IndexOutOfBounds,
                String::from("ElementIndex not smaller than GrB_Index::MAX"),
                None,
            )
            .into())
        }
    }

    fn from_graphblas_index(index: GrB_Index) -> Result<Self, SparseLinearAlgebraError> {
        if index < (ElementIndex::MAX as GrB_Index) {
            // as u64 will truncate to u64::MAX if usize::MAX would be larger than u64::MAX
            Ok(index as ElementIndex)
        } else {
            Err(SystemError::new(
                SystemErrorType::IndexOutOfBounds,
                String::from("GraphBLAS index not smaller than ElementIndex::MAX"),
                None,
            )
            .into())
        }
    }
}
