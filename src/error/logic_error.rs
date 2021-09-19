// Look here for an example on how to implement error types: https://doc.rust-lang.org/src/std/io/error.rs.html#42
use std::error;
use std::error::Error;
use std::fmt;

use super::graphblas_error::{GraphBlasError, GraphBlasErrorType};

#[derive(Debug)]
pub struct LogicError {
    error_type: LogicErrorType,
    explanation: String,
    source: Option<LogicErrorSource>,
}

#[derive(Debug)]
pub enum LogicErrorSource {
    GraphBlas(GraphBlasError),
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicErrorType {
    GraphBlas(GraphBlasErrorType),
    // CreateGraphBlasErrorOnSuccessValue,
    // UnsupportedGraphBlasErrorValue,
    // UninitialisedContext,
    // ContextAlreadyInitialized,
    IndexOutOfBounds,
    Other,
}

impl LogicError {
    pub fn new(
        error_type: LogicErrorType,
        explanation: String,
        source: Option<LogicErrorSource>,
    ) -> Self {
        Self {
            error_type,
            explanation,
            source,
        }
    }

    pub fn error_type(&self) -> LogicErrorType {
        self.error_type.clone()
    }
    pub fn explanation(&self) -> String {
        self.explanation.clone()
    }
}

impl error::Error for LogicError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self.source {
            Some(ref error) => match error {
                LogicErrorSource::GraphBlas(error) => Some(error),
            },
            None => None,
        }
    }
}

impl fmt::Display for LogicError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.error_type {
            // LogicErrorType::GraphBlas(_err) => writeln!(f, "Context:\n{}", &self.context)?,
            _ => writeln!(f, "Explanation:\n{}", &self.explanation)?,
        };

        match &self.source() {
            Some(err) => writeln!(f, "Source error:\n{}", err)?,
            &None => (),
        }
        Ok(())
    }
}

impl From<GraphBlasError> for LogicError {
    fn from(error: GraphBlasError) -> Self {
        Self {
            error_type: LogicErrorType::GraphBlas(error.error_type()),
            explanation: String::new(),
            source: Some(LogicErrorSource::GraphBlas(error)),
        }
    }
}
