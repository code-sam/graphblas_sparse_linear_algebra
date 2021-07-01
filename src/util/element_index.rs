use crate::bindings_to_graphblas_implementation::GrB_Index;
use crate::error::{SystemError, SystemErrorType};

pub type ElementIndex = usize;

pub trait IndexConversion {
    fn to_graphblas_index(&self) -> Result<GrB_Index, SystemError>;
    fn as_graphblas_index(self) -> Result<GrB_Index, SystemError>;
    fn from_graphblas_index(index: GrB_Index) -> Result<ElementIndex, SystemError>;
    // fn to_usize(&self) -> Result<usize, SystemError>;
}

impl IndexConversion for ElementIndex {
    fn to_graphblas_index(&self) -> Result<GrB_Index, SystemError> {
        if *self < (GrB_Index::MAX as Self) {
            // as u64 will truncate to u64::MAX if usize::MAX would be larger than u64::MAX
            Ok(self.clone() as GrB_Index)
        } else {
            Err(SystemError::new(
                SystemErrorType::IndexOutOfBounds,
                String::from("ElementIndex not smaller than GrB_Index::MAX"),
                None,
            ))
        }
    }

    fn as_graphblas_index(self) -> Result<GrB_Index, SystemError> {
        if self < (GrB_Index::MAX as Self) {
            // as u64 will truncate to u64::MAX if usize::MAX would be larger than u64::MAX
            Ok(self as GrB_Index)
        } else {
            Err(SystemError::new(
                SystemErrorType::IndexOutOfBounds,
                String::from("ElementIndex not smaller than GrB_Index::MAX"),
                None,
            ))
        }
    }

    fn from_graphblas_index(index: GrB_Index) -> Result<Self, SystemError> {
        if index < (ElementIndex::MAX as GrB_Index) {
            // as u64 will truncate to u64::MAX if usize::MAX would be larger than u64::MAX
            Ok(index as ElementIndex)
        } else {
            Err(SystemError::new(
                SystemErrorType::IndexOutOfBounds,
                String::from("GraphBLAS index not smaller than ElementIndex::MAX"),
                None,
            ))
        }
    }
}
