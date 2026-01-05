use once_cell::sync::Lazy;
use suitesparse_graphblas_sys::{
    GxB_Iterator, GxB_Iterator_free, GxB_Vector_Iterator_attach, GxB_Vector_Iterator_next,
    GxB_Vector_Iterator_seek,
};

use crate::collections::{new_graphblas_iterator, GetElementValueAtIteratorPosition};
use crate::context::CallGraphBlasContext;
use crate::error::{GraphblasErrorType, LogicErrorType, SparseLinearAlgebraErrorType};
use crate::operators::options::GetGraphblasDescriptor;
use crate::{
    collections::sparse_vector::{GetGraphblasSparseVector, SparseVector},
    context::GetContext,
    error::SparseLinearAlgebraError,
    operators::options::OperatorOptions,
    value_type::ValueType,
};

static DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS: Lazy<OperatorOptions> =
    Lazy::new(|| OperatorOptions::new_default());

pub struct VectorElementValueIterator<'a, T: ValueType + GetElementValueAtIteratorPosition<T>> {
    vector: &'a SparseVector<T>,
    graphblas_iterator: GxB_Iterator,
    next_element: fn(&SparseVector<T>, GxB_Iterator) -> Option<T>,
}

impl<'a, T: ValueType + GetElementValueAtIteratorPosition<T>> VectorElementValueIterator<'a, T> {
    pub fn new(vector: &'a SparseVector<T>) -> Result<Self, SparseLinearAlgebraError> {
        let graphblas_iterator = unsafe { new_graphblas_iterator(vector.context_ref()) }?;

        Ok(Self {
            vector,
            graphblas_iterator,
            next_element: initial_vector_element_value,
        })
    }
}

fn initial_vector_element_value<T: ValueType + GetElementValueAtIteratorPosition<T>>(
    vector: &SparseVector<T>,
    graphblas_iterator: GxB_Iterator,
) -> Option<T> {
    match vector.context_ref().call(
        || unsafe {
            GxB_Vector_Iterator_attach(
                graphblas_iterator,
                vector.graphblas_vector_ptr(),
                DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
            )
        },
        unsafe { &vector.graphblas_vector_ptr() }, // TODO: check that error indeed link to the vector the iterator was attached to
    ) {
        Ok(_) => {}
        Err(error) => return match_iterator_error(error),
    }

    match vector.context_ref().call(
        || unsafe { GxB_Vector_Iterator_seek(graphblas_iterator, 0) },
        unsafe { &vector.graphblas_vector_ptr() }, // TODO: check that error indeed link to the vector the iterator was attached to
    ) {
        Ok(_) => {}
        // TODO: attaching may actually fail, this will cause a panic, which is not desired
        Err(error) => return match_iterator_error(error),
    }

    let next_value = match vector.context_ref().call(
        || unsafe { GxB_Vector_Iterator_seek(graphblas_iterator, 0) },
        unsafe { &vector.graphblas_vector_ptr() }, // TODO: check that error indeed link to the vector the iterator was attached to
    ) {
        Ok(_) => vector_element_value_at_iterator_position(graphblas_iterator),
        Err(error) => match_iterator_error(error),
    };

    return next_value;
}

impl<'a, T: ValueType + GetElementValueAtIteratorPosition<T>> Drop
    for VectorElementValueIterator<'a, T>
{
    fn drop(&mut self) {
        let context = self.vector.context_ref();
        let _ = context.call_without_detailed_error_information(|| unsafe {
            GxB_Iterator_free(&mut self.graphblas_iterator)
        });
    }
}

impl<'a, T: ValueType + GetElementValueAtIteratorPosition<T>> Iterator
    for VectorElementValueIterator<'a, T>
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let next_vector_element_value = (self.next_element)(self.vector, self.graphblas_iterator);

        self.next_element = next_element_value;

        return next_vector_element_value;
    }
}

fn next_element_value<T: ValueType + GetElementValueAtIteratorPosition<T>>(
    vector: &SparseVector<T>,
    graphblas_iterator: GxB_Iterator,
) -> Option<T> {
    match vector.context_ref().call(
        || unsafe { GxB_Vector_Iterator_next(graphblas_iterator) },
        unsafe { &vector.graphblas_vector_ptr() }, // TODO: check that error indeed link to the vector the iterator was attached to
    ) {
        Ok(_) => vector_element_value_at_iterator_position::<T>(graphblas_iterator),
        Err(error) => match_iterator_error(error),
    }
}

fn vector_element_value_at_iterator_position<
    T: ValueType + GetElementValueAtIteratorPosition<T>,
>(
    graphblas_iterator: GxB_Iterator,
) -> Option<T> {
    let element_value = T::element_value_at_iterator_position(graphblas_iterator).unwrap();

    Some(element_value)
}

// REVIEW: always returning None, instead of matching the error, may be more reliable and performant
fn match_iterator_error<T: ValueType>(error: SparseLinearAlgebraError) -> Option<T> {
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

    use crate::collections::sparse_vector::operations::FromVectorElementList;
    use crate::context::Context;
    use crate::operators::binary_operator::First;

    use crate::collections::sparse_vector::{SparseVector, VectorElementList};

    #[test]
    fn test_vector_element_iterator() {
        let context = Context::init_default().unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (0, 1).into(),
            (4, 2).into(),
            (9, 3).into(),
            (16, 4).into(),
        ]);

        let vector_length: usize = 20;
        let vector = SparseVector::<u8>::from_element_list(
            context.clone(),
            vector_length,
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        let vector_element_iterator = VectorElementValueIterator::new(&vector).unwrap();

        for (element_value, expected_element_value) in vector_element_iterator
            .into_iter()
            .zip(element_list.values_ref())
        {
            assert_eq!(element_value, *expected_element_value);
        }
    }

    #[test]
    fn test_iterate_over_empty_vector() {
        let context = Context::init_default().unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![]);

        let vector_length: usize = 20;
        let vector = SparseVector::<u8>::from_element_list(
            context.clone(),
            vector_length,
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        let vector_element_iterator = VectorElementValueIterator::new(&vector).unwrap();

        for (element_value, expected_element_value) in vector_element_iterator
            .into_iter()
            .zip(element_list.values_ref())
        {
            assert_eq!(element_value, *expected_element_value);
        }
    }
}
