// Look here for an example on how to implement error types: https://doc.rust-lang.org/src/std/io/error.rs.html#42
use std::error::Error;
use std::fmt;
use std::{error, num::TryFromIntError};

use super::graphblas_error::{GraphblasError, GraphblasErrorType};

#[derive(Debug)]
pub struct LogicError {
    error_type: LogicErrorType,
    explanation: String,
    source: Option<LogicErrorSource>,
}

#[derive(Debug)]
pub enum LogicErrorSource {
    GraphBlas(GraphblasError),
    TryFromIntError(TryFromIntError),
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicErrorType {
    GraphBlas(GraphblasErrorType),
    // CreateGraphBlasErrorOnSuccessValue,
    // UnsupportedGraphBlasErrorValue,
    // UninitialisedContext,
    // ContextAlreadyInitialized,
    IndexOutOfBounds,
    UnsafeTypeConversion,
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
        self.error_type.to_owned()
    }
    pub fn explanation(&self) -> String {
        self.explanation.to_owned()
    }
}

impl error::Error for LogicError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self.source {
            Some(ref error) => match error {
                LogicErrorSource::GraphBlas(error) => Some(error),
                LogicErrorSource::TryFromIntError(error) => Some(error),
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

impl From<GraphblasError> for LogicError {
    fn from(error: GraphblasError) -> Self {
        Self {
            error_type: LogicErrorType::GraphBlas(error.error_type()),
            explanation: String::new(),
            source: Some(LogicErrorSource::GraphBlas(error)),
        }
    }
}

impl From<TryFromIntError> for LogicError {
    fn from(error: TryFromIntError) -> Self {
        Self {
            error_type: LogicErrorType::UnsafeTypeConversion,
            explanation: String::new(),
            source: Some(LogicErrorSource::TryFromIntError(error)),
        }
    }
}
