use std::sync::Arc;

use once_cell::sync::Lazy;
use suitesparse_graphblas_sys::{
    GrB_Matrix, GxB_Iterator, GxB_Iterator_free, GxB_Matrix_Iterator_attach, GxB_Matrix_Iterator_next, GxB_Matrix_Iterator_seek
};

use crate::collections::sparse_matrix::MatrixElement;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::{new_graphblas_iterator, GetElementValueAtIteratorPosition};
use crate::context::{CallGraphBlasContext, Context};
use crate::context::GetContext;
use crate::error::SparseLinearAlgebraError;
use crate::error::{GraphblasErrorType, LogicErrorType, SparseLinearAlgebraErrorType};
use crate::operators::options::GetGraphblasDescriptor;
use crate::operators::options::OperatorOptions;
use crate::value_type::ValueType;

use super::matrix_element_coordinate_at_iterator_position;

static DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS: Lazy<OperatorOptions> =
    Lazy::new(|| OperatorOptions::new_default());

pub struct MatrixElementIterator<'a, T: ValueType + GetElementValueAtIteratorPosition<T>> {
    graphblas_context: Arc<Context>,
    graphblas_matrix: &'a GrB_Matrix,
    graphblas_iterator: GxB_Iterator,
    next_element: fn(&Arc<Context>, &GrB_Matrix, GxB_Iterator) -> Option<MatrixElement<T>>,
}

impl<'a, T: ValueType + GetElementValueAtIteratorPosition<T>> MatrixElementIterator<'a, T> {
    pub fn new(graphblas_matrix: &'a (impl GetGraphblasSparseMatrix + GetContext)) -> Result<Self, SparseLinearAlgebraError> {
        let graphblas_iterator = unsafe { new_graphblas_iterator(graphblas_matrix.context_ref()) }?;

        Ok(Self {
            graphblas_context: graphblas_matrix.context(),
            graphblas_matrix: unsafe { graphblas_matrix.graphblas_matrix_ref() },
            graphblas_iterator,
            next_element: initial_matrix_element,
        })
    }
}

fn initial_matrix_element<T: ValueType + GetElementValueAtIteratorPosition<T>>(
    graphblas_context: &Arc<Context>,
    graphblas_matrix: &GrB_Matrix,
    graphblas_iterator: GxB_Iterator,
) -> Option<MatrixElement<T>> {
    match graphblas_context.call(
        || unsafe {
            GxB_Matrix_Iterator_attach(
                graphblas_iterator,
                graphblas_matrix.to_owned(),
                DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
            )
        },
        graphblas_matrix, // TODO: check that error indeed link to the matrix the iterator was attached to
    ) {
        Ok(_) => {}
        Err(error) => return match_iterator_error(error),
    }

    match graphblas_context.call(
        || unsafe { GxB_Matrix_Iterator_seek(graphblas_iterator, 0) },
        graphblas_matrix, // TODO: check that error indeed link to the matrix the iterator was attached to
    ) {
        Ok(_) => {}
        // TODO: attaching may actually fail, this will cause a panic, which is not desired
        Err(error) => return match_iterator_error(error),
    }

    let next_value = match graphblas_context.call(
        || unsafe { GxB_Matrix_Iterator_seek(graphblas_iterator, 0) },
        graphblas_matrix, // TODO: check that error indeed link to the matrix the iterator was attached to
    ) {
        Ok(_) => matrix_element_at_iterator_position(graphblas_iterator),
        Err(error) => match_iterator_error(error),
    };

    next_value
}

impl<'a, T: ValueType + GetElementValueAtIteratorPosition<T>> Drop
    for MatrixElementIterator<'a, T>
{
    fn drop(&mut self) {
        let _ = self.graphblas_context.call_without_detailed_error_information(|| unsafe {
            GxB_Iterator_free(&mut self.graphblas_iterator)
        });
    }
}

impl<'a, T: ValueType + GetElementValueAtIteratorPosition<T>> Iterator
    for MatrixElementIterator<'a, T>
{
    type Item = MatrixElement<T>;

    fn next(&mut self) -> Option<MatrixElement<T>> {
        let next_matrix_element = (self.next_element)(&self.graphblas_context, self.graphblas_matrix, self.graphblas_iterator);

        self.next_element = next_element;

        return next_matrix_element;
    }
}

fn next_element<T: ValueType + GetElementValueAtIteratorPosition<T>>(
    graphblas_context: &Arc<Context>,
    graphblas_matrix: &GrB_Matrix,
    graphblas_iterator: GxB_Iterator,
) -> Option<MatrixElement<T>> {
    match graphblas_context.call(
        || unsafe { GxB_Matrix_Iterator_next(graphblas_iterator) },
        graphblas_matrix, // TODO: check that error indeed link to the matrix the iterator was attached to
    ) {
        Ok(_) => matrix_element_at_iterator_position::<T>(graphblas_iterator),
        Err(error) => match_iterator_error(error),
    }
}

fn matrix_element_at_iterator_position<T: ValueType + GetElementValueAtIteratorPosition<T>>(
    graphblas_iterator: GxB_Iterator,
) -> Option<MatrixElement<T>> {
    let element_coordinate =
        matrix_element_coordinate_at_iterator_position(graphblas_iterator).unwrap();

    let element_value = T::element_value_at_iterator_position(graphblas_iterator).unwrap();

    Some(MatrixElement::new(element_coordinate, element_value))
}

// REVIEW: always returning None, instead of matching the error, may be more reliable and performant
fn match_iterator_error<T: ValueType>(error: SparseLinearAlgebraError) -> Option<MatrixElement<T>> {
    #[cfg(debug_assertions)]
    match error.error_type() {
        SparseLinearAlgebraErrorType::LogicErrorType(error_type) => match error_type {
            LogicErrorType::GraphBlas(graphblas_error) => match graphblas_error {
                GraphblasErrorType::IteratorExhausted => None,
                _ => panic!("An unexpected error occured while iterating: {}", error),
            },
            _ => panic!("An unexpected error occured while iterating: {}", error),
        },
        _ => panic!("An unexpected error occured while iterating: {}", error),
    }

    #[cfg(not(debug_assertions))]
    None
}

// pub trait IterateOverMatrix: Iterator {}

// impl<'a, T: ValueType> IterateOverMatrix for MatrixElementIterator<'a, T> {

// }

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::FromMatrixElementList;
    use crate::context::Context;
    use crate::operators::binary_operator::First;

    use crate::collections::sparse_matrix::{
        GetMatrixElementCoordinate, GetMatrixElementValue, MatrixElementList, Size, SparseMatrix,
    };

    #[test]
    fn test_matrix_element_iterator() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 1, 1).into(),
            (4, 2, 2).into(),
            (9, 3, 3).into(),
            (16, 4, 4).into(),
        ]);

        let matrix_size: Size = (20, 20).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            matrix_size,
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        let matrix_element_iterator = MatrixElementIterator::<u8>::new(&matrix).unwrap();

        for (element, expected_element) in matrix_element_iterator
            .into_iter()
            .zip(element_list.matrix_elements().into_iter())
        {
            assert_eq!(element.coordinate_ref(), expected_element.coordinate_ref());
            assert_eq!(element.value_ref(), expected_element.value_ref());
        }
    }

    #[test]
    fn test_iterate_over_empty_matrix() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![]);

        let matrix_size: Size = (20, 20).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            matrix_size,
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        let matrix_element_iterator = MatrixElementIterator::<u8>::new(&matrix).unwrap();

        for (element, expected_element) in matrix_element_iterator
            .into_iter()
            .zip(element_list.matrix_elements().into_iter())
        {
            assert_eq!(element.coordinate_ref(), expected_element.coordinate_ref());
            assert_eq!(element.value_ref(), expected_element.value_ref());
        }
    }
}
