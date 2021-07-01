// Look here for an example on how to implement error types: https://doc.rust-lang.org/src/std/io/error.rs.html#42
use std::error;
use std::error::Error;
use std::fmt;
// use std::num::TryFromIntError;

use super::graphblas_error::{GraphBlasError, GraphBlasErrorType};

#[derive(Debug)]
pub struct SystemError {
    error_type: SystemErrorType,
    context: String,
    source: Option<SystemErrorSource>,
}

#[derive(Debug)]
pub enum SystemErrorSource {
    GraphBlas(GraphBlasError),
}

// impl error::Error for SystemErrorSource {
//     fn source(&self) ->  Option<&(dyn error::Error + 'static)> {
//         match self {
//             SystemErrorSource::GraphBlas(error) => Some(error)
//         }
//     }
// }

// impl fmt::Display for SystemErrorSource {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         match self {
//             SystemErrorSource::GraphBlas(error) => {writeln!(f, "{}", error);}
//         }
//         Ok(())
//     }
// }

#[derive(Debug, Clone, PartialEq)]
pub enum SystemErrorType {
    GraphBlas(GraphBlasErrorType),
    CreateGraphBlasErrorOnSuccessValue,
    UnsupportedGraphBlasErrorValue,
    UninitialisedContext,
    ContextAlreadyInitialized,
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
                SystemErrorSource::GraphBlas(error) => Some(error),
            },
            None => None,
        }
    }
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.error_type {
            // LogicErrorType::GraphBlas(_err) => writeln!(f, "Context:\n{}", &self.context)?,
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
            error_type: SystemErrorType::GraphBlas(error.error_type()),
            context: String::new(),
            source: Some(SystemErrorSource::GraphBlas(error)),
        }
    }
}

// impl From<TryFromIntError> for SystemError {
//     fn from(error: TryFromIntError) -> Self {
//         Self {
//             error_type: SystemErrorType::IndexOutOfBounds,
//             context: format!("{}", error),
//         }
//     }
// }
