use std::ptr;

use once_cell::sync::Lazy;

use suitesparse_graphblas_sys::GxB_Vector_sort;

use crate::collections::sparse_vector::SparseVectorTrait;
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::index::ElementIndex;
use crate::operators::options::OperatorOptions;
use crate::{
    collections::sparse_vector::{GraphblasSparseVectorTrait, SparseVector},
    error::SparseLinearAlgebraError,
    operators::binary_operator::{BinaryOperator, ReturnsBool},
    value_type::ValueType,
};

static DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS: Lazy<OperatorOptions> =
    Lazy::new(|| OperatorOptions::new_default());

// REVIEW: support typecasting for indices and the evaluation domain of the binary operator
pub trait SortSparseVector<T: ValueType, B: BinaryOperator<T, T, bool, T>> {
    fn sort(&mut self, sort_operator: &B) -> Result<(), SparseLinearAlgebraError>;

    fn sorted_values_and_indices(
        &self,
        sorted_values: &mut SparseVector<T>,
        indices_to_sort_self: &mut SparseVector<ElementIndex>,
        sort_operator: &B,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn sorted_values(&self, sort_operator: &B)
        -> Result<SparseVector<T>, SparseLinearAlgebraError>;

    fn indices_to_sort(
        &self,
        sort_operator: &B,
    ) -> Result<SparseVector<ElementIndex>, SparseLinearAlgebraError>;
}

impl<T: ValueType, B: BinaryOperator<T, T, bool, T> + ReturnsBool> SortSparseVector<T, B>
    for SparseVector<T>
{
    fn sort(&mut self, sort_operator: &B) -> Result<(), SparseLinearAlgebraError> {
        self.context_ref().call(
            || unsafe {
                GxB_Vector_sort(
                    self.graphblas_vector(),
                    ptr::null_mut(),
                    sort_operator.graphblas_type(),
                    self.graphblas_vector(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            unsafe { self.graphblas_vector_ref() },
        )?;
        Ok(())
    }

    fn sorted_values_and_indices(
        &self,
        sorted_values: &mut SparseVector<T>,
        indices_to_sort_self: &mut SparseVector<ElementIndex>,
        sort_operator: &B,
    ) -> Result<(), SparseLinearAlgebraError> {
        self.context_ref().call(
            || unsafe {
                GxB_Vector_sort(
                    sorted_values.graphblas_vector(),
                    indices_to_sort_self.graphblas_vector(),
                    sort_operator.graphblas_type(),
                    self.graphblas_vector(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            unsafe { self.graphblas_vector_ref() },
        )?;
        Ok(())
    }

    fn sorted_values(
        &self,
        sort_operator: &B,
    ) -> Result<SparseVector<T>, SparseLinearAlgebraError> {
        let sorted_values = SparseVector::<T>::new(self.context_ref(), &self.length()?)?;
        self.context_ref().call(
            || unsafe {
                GxB_Vector_sort(
                    sorted_values.graphblas_vector(),
                    ptr::null_mut(),
                    sort_operator.graphblas_type(),
                    self.graphblas_vector(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            unsafe { self.graphblas_vector_ref() },
        )?;
        Ok(sorted_values)
    }

    fn indices_to_sort(
        &self,
        sort_operator: &B,
    ) -> Result<SparseVector<ElementIndex>, SparseLinearAlgebraError> {
        let mut indices_to_sort_self =
            SparseVector::<ElementIndex>::new(self.context_ref(), &self.length()?)?;
        self.context_ref().call(
            || unsafe {
                GxB_Vector_sort(
                    ptr::null_mut(),
                    indices_to_sort_self.graphblas_vector(),
                    sort_operator.graphblas_type(),
                    self.graphblas_vector(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            unsafe { self.graphblas_vector_ref() },
        )?;
        Ok(indices_to_sort_self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, SparseVectorTrait,
    };
    use crate::operators::binary_operator::{First, IsGreaterThan};
    use crate::{
        collections::sparse_vector::{SparseVector, VectorElementList},
        context::{Context, Mode},
    };

    #[test]
    fn sorted_values_and_indices() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<isize>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 6).into(),
            (6, 4).into(),
        ]);

        let vector = SparseVector::<isize>::from_element_list(
            &context.clone(),
            &10,
            &element_list,
            &First::<isize, isize, isize, isize>::new(),
        )
        .unwrap();

        let mut sorted = SparseVector::new(&context, &vector.length().unwrap()).unwrap();
        let mut indices = SparseVector::new(&context, &vector.length().unwrap()).unwrap();

        let larger_than_operator = IsGreaterThan::<isize, isize, bool, isize>::new();

        vector
            .sorted_values_and_indices(&mut sorted, &mut indices, &larger_than_operator)
            .unwrap();

        assert_eq!(sorted.get_element_value_or_default(&0).unwrap(), 6);
        assert_eq!(sorted.get_element_value_or_default(&1).unwrap(), 4);
        assert_eq!(sorted.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(sorted.get_element_value_or_default(&3).unwrap(), 1);

        assert_eq!(indices.get_element_value_or_default(&0).unwrap(), 4);
        assert_eq!(indices.get_element_value_or_default(&1).unwrap(), 6);
        assert_eq!(indices.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(indices.get_element_value_or_default(&3).unwrap(), 1);
    }

    #[test]
    fn sort() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<isize>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 6).into(),
            (6, 4).into(),
        ]);

        let mut vector = SparseVector::<isize>::from_element_list(
            &context.clone(),
            &10,
            &element_list,
            &First::<isize, isize, isize, isize>::new(),
        )
        .unwrap();

        let larger_than_operator = IsGreaterThan::<isize, isize, bool, isize>::new();

        vector.sort(&larger_than_operator).unwrap();

        assert_eq!(vector.get_element_value_or_default(&0).unwrap(), 6);
        assert_eq!(vector.get_element_value_or_default(&1).unwrap(), 4);
        assert_eq!(vector.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(vector.get_element_value_or_default(&3).unwrap(), 1);
    }

    #[test]
    fn sorted_values() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<isize>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 6).into(),
            (6, 4).into(),
        ]);

        let vector = SparseVector::<isize>::from_element_list(
            &context.clone(),
            &10,
            &element_list,
            &First::<isize, isize, isize, isize>::new(),
        )
        .unwrap();

        let larger_than_operator = IsGreaterThan::<isize, isize, bool, isize>::new();

        let sorted = vector.sorted_values(&larger_than_operator).unwrap();

        assert_eq!(sorted.get_element_value_or_default(&0).unwrap(), 6);
        assert_eq!(sorted.get_element_value_or_default(&1).unwrap(), 4);
        assert_eq!(sorted.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(sorted.get_element_value_or_default(&3).unwrap(), 1);
    }

    #[test]
    fn indices_to_sort() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<isize>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 6).into(),
            (6, 4).into(),
        ]);

        let vector = SparseVector::<isize>::from_element_list(
            &context.clone(),
            &10,
            &element_list,
            &First::<isize, isize, isize, isize>::new(),
        )
        .unwrap();

        let larger_than_operator = IsGreaterThan::<isize, isize, bool, isize>::new();

        let indices = vector.indices_to_sort(&larger_than_operator).unwrap();

        assert_eq!(indices.get_element_value_or_default(&0).unwrap(), 4);
        assert_eq!(indices.get_element_value_or_default(&1).unwrap(), 6);
        assert_eq!(indices.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(indices.get_element_value_or_default(&3).unwrap(), 1);
    }
}
