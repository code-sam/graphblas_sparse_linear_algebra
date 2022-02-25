use std::error;
use std::error::Error;
use std::fmt;
use std::num::TryFromIntError;

use super::graphblas_error::{GraphBlasError, GraphBlasErrorType};

#[derive(Debug)]
pub struct SystemError {
    error_type: SystemErrorType,
    explanation: String,
    source: Option<SystemErrorSource>,
}

#[derive(Debug)]
pub enum SystemErrorSource {
    GraphBLAS(GraphBlasError),
    IntegerConversionError(TryFromIntError),
    PoisonedData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemErrorType {
    GraphBLAS(GraphBlasErrorType),
    CreateGraphBlasErrorOnSuccessValue,
    ContextAlreadyInitialized,
    IndexOutOfBounds,
    UninitialisedContext,
    UnsupportedGraphBlasErrorValue,
    PoisonedData,
    IntegerConversionFailed,
    Other,
}

impl SystemError {
    pub fn new(
        error_type: SystemErrorType,
        explanation: String,
        source: Option<SystemErrorSource>,
    ) -> Self {
        Self {
            error_type,
            explanation,
            source,
        }
    }

    pub fn error_type(&self) -> SystemErrorType {
        self.error_type.clone()
    }
    pub fn explanation(&self) -> String {
        self.explanation.clone()
    }
}

impl error::Error for SystemError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self.source {
            Some(ref error) => match error {
                SystemErrorSource::GraphBLAS(error) => Some(error),
                SystemErrorSource::IntegerConversionError(error) => Some(error),
                SystemErrorSource::PoisonedData => None,
            },
            None => None,
        }
    }
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.error_type {
            _ => writeln!(f, "Explanation:\n{}", &self.explanation)?,
        };

        match &self.source() {
            Some(err) => writeln!(f, "Source error:\n{}", err)?,
            None => (),
        }
        Ok(())
    }
}

impl From<GraphBlasError> for SystemError {
    fn from(error: GraphBlasError) -> Self {
        Self {
            error_type: SystemErrorType::GraphBLAS(error.error_type()),
            explanation: String::new(),
            source: Some(SystemErrorSource::GraphBLAS(error)),
        }
    }
}

impl From<TryFromIntError> for SystemError {
    fn from(error: TryFromIntError) -> Self {
        Self {
            error_type: SystemErrorType::IntegerConversionFailed,
            explanation: String::new(),
            source: Some(SystemErrorSource::IntegerConversionError(error)),
        }
    }
}
