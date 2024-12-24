use std::mem::MaybeUninit;

use once_cell::sync::Lazy;
use suitesparse_graphblas_sys::{
    GxB_Iterator, GxB_Iterator_free, GxB_Matrix_Iterator_attach, GxB_Matrix_Iterator_getIndex,
    GxB_Matrix_Iterator_next, GxB_Matrix_Iterator_seek,
};

use crate::collections::new_graphblas_iterator;
use crate::collections::sparse_matrix::{Coordinate, GetGraphblasSparseMatrix, SparseMatrix};
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::error::SparseLinearAlgebraError;
use crate::error::{GraphblasErrorType, LogicErrorType, SparseLinearAlgebraErrorType};
use crate::index::ElementIndex;
use crate::index::IndexConversion;
use crate::operators::options::GetGraphblasDescriptor;
use crate::operators::options::OperatorOptions;
use crate::value_type::ValueType;

static DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS: Lazy<OperatorOptions> =
    Lazy::new(|| OperatorOptions::new_default());

pub struct MatrixElementCoordinateIterator<'a, T: ValueType> {
    matrix: &'a SparseMatrix<T>,
    graphblas_iterator: GxB_Iterator,
    next_element: fn(&SparseMatrix<T>, GxB_Iterator) -> Option<Coordinate>,
}

impl<'a, T: ValueType> MatrixElementCoordinateIterator<'a, T> {
    pub fn new(matrix: &'a SparseMatrix<T>) -> Result<Self, SparseLinearAlgebraError> {
        let graphblas_iterator = unsafe { new_graphblas_iterator(matrix.context_ref()) }?;

        Ok(Self {
            matrix,
            graphblas_iterator,
            next_element: initial_matrix_element_coordinate,
        })
    }
}

fn initial_matrix_element_coordinate<T: ValueType>(
    matrix: &SparseMatrix<T>,
    graphblas_iterator: GxB_Iterator,
) -> Option<Coordinate> {
    match matrix.context_ref().call(
        || unsafe {
            GxB_Matrix_Iterator_attach(
                graphblas_iterator,
                matrix.graphblas_matrix(),
                DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
            )
        },
        unsafe { &matrix.graphblas_matrix() }, // TODO: check that error indeed link to the matrix the iterator was attached to
    ) {
        Ok(_) => {}
        Err(error) => return match_iterator_error(error),
    }

    match matrix.context_ref().call(
        || unsafe { GxB_Matrix_Iterator_seek(graphblas_iterator, 0) },
        unsafe { &matrix.graphblas_matrix() }, // TODO: check that error indeed link to the matrix the iterator was attached to
    ) {
        Ok(_) => {}
        // TODO: attaching may actually fail, this will cause a panic, which is not desired
        Err(error) => return match_iterator_error(error),
    }

    let next_index = match matrix.context_ref().call(
        || unsafe { GxB_Matrix_Iterator_seek(graphblas_iterator, 0) },
        unsafe { &matrix.graphblas_matrix() }, // TODO: check that error indeed link to the matrix the iterator was attached to
    ) {
        Ok(_) => matrix_element_coordinate_at_iterator_position(graphblas_iterator),
        Err(error) => match_iterator_error(error),
    };

    return next_index;
}

impl<'a, T: ValueType> Drop for MatrixElementCoordinateIterator<'a, T> {
    fn drop(&mut self) {
        let context = self.matrix.context_ref();
        let _ = context.call_without_detailed_error_information(|| unsafe {
            GxB_Iterator_free(&mut self.graphblas_iterator)
        });
    }
}

impl<'a, T: ValueType> Iterator for MatrixElementCoordinateIterator<'a, T> {
    type Item = Coordinate;

    fn next(&mut self) -> Option<Coordinate> {
        let next_matrix_element_coordinate =
            (self.next_element)(self.matrix, self.graphblas_iterator);

        self.next_element = next_element_coordinate;

        return next_matrix_element_coordinate;
    }
}

fn next_element_coordinate<T: ValueType>(
    matrix: &SparseMatrix<T>,
    graphblas_iterator: GxB_Iterator,
) -> Option<Coordinate> {
    match matrix.context_ref().call(
        || unsafe { GxB_Matrix_Iterator_next(graphblas_iterator) },
        unsafe { &matrix.graphblas_matrix() }, // TODO: check that error indeed link to the matrix the iterator was attached to
    ) {
        Ok(_) => matrix_element_coordinate_at_iterator_position(graphblas_iterator),
        Err(error) => match_iterator_error(error),
    }
}

pub(super) fn matrix_element_coordinate_at_iterator_position(
    graphblas_iterator: GxB_Iterator,
) -> Option<Coordinate> {
    let mut row_index = MaybeUninit::uninit();
    let mut column_index = MaybeUninit::uninit();

    unsafe {
        GxB_Matrix_Iterator_getIndex(
            graphblas_iterator,
            row_index.as_mut_ptr(),
            column_index.as_mut_ptr(),
        )
    };

    let row_index = unsafe { row_index.assume_init() };
    let column_index = unsafe { column_index.assume_init() };

    let element_coordinate = Coordinate::new(
        ElementIndex::from_graphblas_index(row_index).unwrap(),
        ElementIndex::from_graphblas_index(column_index).unwrap(),
    );

    Some(element_coordinate)
}

// REVIEW: always returning None, instead of matching the error, may be more reliable and performant
fn match_iterator_error(error: SparseLinearAlgebraError) -> Option<Coordinate> {
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

// pub trait IterateOverVector: Iterator {}

// impl<'a, T: ValueType> IterateOverVector for VectorElementIterator<'a, T> {

// }

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::FromMatrixElementList;
    use crate::context::Context;
    use crate::operators::binary_operator::First;

    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};

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

        let matrix_element_index_iterator = MatrixElementCoordinateIterator::new(&matrix).unwrap();

        for (element_coordinate, expected_element_coordinate) in matrix_element_index_iterator
            .into_iter()
            .zip(element_list.coordinates().into_iter())
        {
            assert_eq!(element_coordinate, expected_element_coordinate);
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

        let matrix_element_iterator = MatrixElementCoordinateIterator::new(&matrix).unwrap();

        for (element_coordinate, expected_element_coordinate) in matrix_element_iterator
            .into_iter()
            .zip(element_list.coordinates().into_iter())
        {
            assert_eq!(element_coordinate, expected_element_coordinate);
        }
    }
}
