// Look here for an example on how to implement error types: https://doc.rust-lang.org/src/std/io/error.rs.html#42
use std::error;
use std::fmt;
// use std::io;
use std::error::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct GraphblasError {
    error_type: GraphblasErrorType,
    explanation: String,
    // source: Option<Box<(dyn error::Error)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GraphblasErrorType {
    NoValue,
    UnitializedObject,
    InvalidObject,
    NotImplemented,
    NullPointer,
    InvalidValue,
    InvalidIndex,
    DomainMismatch,
    DimensionMismatch,
    EmptyObject,
    OutputNotEmpty,
    OutOfMemory,
    InsufficientSpace,
    IndexOutOfBounds,
    IteratorExhausted,
    Panic,
}

impl GraphblasError {
    pub fn new(error_type: GraphblasErrorType, explanation: String) -> Self {
        Self {
            error_type: error_type,
            explanation: explanation,
        }
    }

    pub fn error_type(&self) -> GraphblasErrorType {
        self.error_type.to_owned()
    }
    pub fn explanation(&self) -> String {
        self.explanation.to_owned()
    }
}

impl error::Error for GraphblasError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.error_type {
            // ErrorType::IO(error) => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for GraphblasError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // REVIEW: is this match useful?
        match &self.error_type {
            // ErrorType::IO(_err) => writeln!(f, "Context:\n{}", &self.context)?,
            _ => writeln!(f, "Explanation:\n{}", &self.explanation)?,
        };

        match &self.source() {
            Some(err) => writeln!(f, "Source error:\n{}", err)?,
            &None => (),
        }
        Ok(())
    }
}
