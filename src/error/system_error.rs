use std::error;
use std::error::Error;
use std::fmt;

use super::graphblas_error::{GraphBlasError, GraphBlasErrorType};

#[derive(Debug)]
pub struct SystemError {
    error_type: SystemErrorType,
    context: String,
    source: Option<SystemErrorSource>,
}

#[derive(Debug)]
pub enum SystemErrorSource {
    GraphBLAS(GraphBlasError),
    PoisonedData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemErrorType {
    GraphBLAS(GraphBlasErrorType),
    CreateGraphBlasErrorOnSuccessValue,
    UnsupportedGraphBlasErrorValue,
    UninitialisedContext,
    ContextAlreadyInitialized,
    PoisonedData,
    IndexOutOfBounds,
    Other,
}

impl SystemError {
    pub fn new(
        error_type: SystemErrorType,
        context: String,
        source: Option<SystemErrorSource>,
    ) -> Self {
        Self {
            error_type,
            context,
            source,
        }
    }

    pub fn error_type(&self) -> SystemErrorType {
        self.error_type.clone()
    }
    pub fn context(&self) -> String {
        self.context.clone()
    }
}

impl error::Error for SystemError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self.source {
            Some(ref error) => match error {
                SystemErrorSource::GraphBLAS(error) => Some(error),
                SystemErrorSource::PoisonedData => None
            },
            None => None,
        }
    }
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.error_type {
            _ => writeln!(f, "Context:\n{}", &self.context)?,
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
            context: String::new(),
            source: Some(SystemErrorSource::GraphBLAS(error)),
        }
    }
}
