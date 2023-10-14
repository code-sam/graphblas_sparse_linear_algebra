use suitesparse_graphblas_sys::GxB_Vector_isStoredElement;

use crate::{
    collections::sparse_vector::{GetGraphblasSparseVector, SparseVector},
    context::{CallGraphBlasContext, GetContext},
    error::{
        GraphblasErrorType, LogicErrorType, SparseLinearAlgebraError, SparseLinearAlgebraErrorType,
    },
    index::{ElementIndex, IndexConversion},
    value_type::ValueType,
};

pub trait IsSparseVectorElement {
    fn is_element(&self, index: ElementIndex) -> Result<bool, SparseLinearAlgebraError>;
    fn try_is_element(&self, index: ElementIndex) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> IsSparseVectorElement for SparseVector<T> {
    fn is_element(&self, index: ElementIndex) -> Result<bool, SparseLinearAlgebraError> {
        let index = index.to_graphblas_index()?;

        match self.context_ref().call(
            || unsafe { GxB_Vector_isStoredElement(self.graphblas_vector(), index) },
            unsafe { &self.graphblas_vector() },
        ) {
            Ok(_) => Ok(true),
            Err(error) => match error.error_type() {
                SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                    GraphblasErrorType::NoValue,
                )) => Ok(false),
                _ => Err(error),
            },
        }
    }

    fn try_is_element(&self, index: ElementIndex) -> Result<(), SparseLinearAlgebraError> {
        let index = index.to_graphblas_index()?;

        match self.context_ref().call(
            || unsafe { GxB_Vector_isStoredElement(self.graphblas_vector(), index) },
            unsafe { &self.graphblas_vector() },
        ) {
            Ok(_) => Ok(()),
            Err(error) => Err(error),
        }
    }
}

pub fn is_element(
    vector: &(impl GetGraphblasSparseVector + GetContext),
    index: ElementIndex,
) -> Result<bool, SparseLinearAlgebraError> {
    let index = index.to_graphblas_index()?;

    match vector.context_ref().call(
        || unsafe { GxB_Vector_isStoredElement(vector.graphblas_vector(), index) },
        unsafe { &vector.graphblas_vector() },
    ) {
        Ok(_) => Ok(true),
        Err(error) => match error.error_type() {
            SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                GraphblasErrorType::NoValue,
            )) => Ok(false),
            _ => Err(error),
        },
    }
}

pub fn try_is_element(
    vector: &(impl GetGraphblasSparseVector + GetContext),
    index: ElementIndex,
) -> Result<(), SparseLinearAlgebraError> {
    let index = index.to_graphblas_index()?;

    match vector.context_ref().call(
        || unsafe { GxB_Vector_isStoredElement(vector.graphblas_vector(), index) },
        unsafe { &vector.graphblas_vector() },
    ) {
        Ok(_) => Ok(()),
        Err(error) => Err(error),
    }
}
