// Look here for an example on how to implement error types: https://doc.rust-lang.org/src/std/io/error.rs.html#42
use super::graphblas_error::{GraphblasError, GraphblasErrorType};
use super::logic_error::{LogicError, LogicErrorType};
use super::other_error::OtherErrorType;
use super::system_error::{SystemError, SystemErrorType};
use std::error;
use std::error::Error;
use std::fmt;

// pub trait SparseLinearAlgebraErrorType {
//     fn error_type(&self) -> Box<dyn Self>;
// }

#[derive(Debug)]
pub enum SparseLinearAlgebraError {
    SystemError(crate::error::SystemError),
    LogicError(crate::error::LogicError),
    OtherError(crate::error::OtherError),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SparseLinearAlgebraErrorType {
    SystemErrorType(SystemErrorType),
    LogicErrorType(LogicErrorType),
    OtherErrorType(OtherErrorType),
}

impl error::Error for SparseLinearAlgebraError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            SparseLinearAlgebraError::SystemError(error) => Some(error),
            SparseLinearAlgebraError::LogicError(error) => Some(error),
            SparseLinearAlgebraError::OtherError(error) => Some(error),
        }
    }
}

impl SparseLinearAlgebraError {
    pub fn error_type(&self) -> SparseLinearAlgebraErrorType {
        match self {
            SparseLinearAlgebraError::SystemError(error) => {
                SparseLinearAlgebraErrorType::SystemErrorType(error.error_type())
            }
            SparseLinearAlgebraError::LogicError(error) => {
                SparseLinearAlgebraErrorType::LogicErrorType(error.error_type())
            }
            SparseLinearAlgebraError::OtherError(error) => {
                SparseLinearAlgebraErrorType::OtherErrorType(error.error_type())
            }
        }
    }
}

impl fmt::Display for SparseLinearAlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", self.source().unwrap());
        Ok(())
    }
}

impl From<SystemError> for SparseLinearAlgebraError {
    fn from(error: SystemError) -> Self {
        SparseLinearAlgebraError::SystemError(error)
    }
}

impl From<LogicError> for SparseLinearAlgebraError {
    fn from(error: LogicError) -> Self {
        SparseLinearAlgebraError::LogicError(error)
    }
}

impl From<std::fmt::Error> for SparseLinearAlgebraError {
    fn from(error: std::fmt::Error) -> Self {
        SparseLinearAlgebraError::OtherError(error.into())
    }
}

impl From<SparseLinearAlgebraError> for std::fmt::Error {
    fn from(_error: SparseLinearAlgebraError) -> Self {
        std::fmt::Error {}
    }
}

impl From<GraphblasError> for SparseLinearAlgebraError {
    fn from(error: GraphblasError) -> Self {
        match error.error_type() {
            GraphblasErrorType::NoValue
            | GraphblasErrorType::UnitializedObject
            | GraphblasErrorType::InvalidObject
            | GraphblasErrorType::NullPointer
            | GraphblasErrorType::InvalidValue
            | GraphblasErrorType::InvalidIndex
            | GraphblasErrorType::DomainMismatch
            | GraphblasErrorType::DimensionMismatch
            | GraphblasErrorType::EmptyObject
            | GraphblasErrorType::OutputNotEmpty
            | GraphblasErrorType::InsufficientSpace
            | GraphblasErrorType::IndexOutOfBounds
            | GraphblasErrorType::IteratorExhausted => Self::LogicError(error.into()),
            GraphblasErrorType::OutOfMemory
            | GraphblasErrorType::Panic
            | GraphblasErrorType::NotImplemented => Self::SystemError(error.into()),
        }
    }
}
