use std::ptr;

use once_cell::sync::Lazy;

use suitesparse_graphblas_sys::{GrB_DESC_T0, GxB_Matrix_sort};

use crate::collections::sparse_matrix::SparseMatrixTrait;
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::index::ElementIndex;
use crate::operators::options::OperatorOptions;
use crate::{
    collections::sparse_matrix::{GraphblasSparseMatrixTrait, SparseMatrix},
    error::SparseLinearAlgebraError,
    operators::binary_operator::{BinaryOperator, ReturnsBool},
    value_type::ValueType,
};

static DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS: Lazy<OperatorOptions> =
    Lazy::new(|| OperatorOptions::new_default());

// REVIEW: support typecasting for indices and the evaluation domain of the binary operator
pub trait SortSparseMatrix<T: ValueType, B: BinaryOperator<T, T, bool, T>> {
    fn sort_rows(&mut self, sort_operator: &B) -> Result<(), SparseLinearAlgebraError>;

    fn sort_columns(&mut self, sort_operator: &B) -> Result<(), SparseLinearAlgebraError>;

    fn sorted_rows_and_indices(
        &self,
        sorted_values: &mut SparseMatrix<T>,
        indices_to_sort_rows: &mut SparseMatrix<ElementIndex>,
        sort_operator: &B,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn sorted_columns_and_indices(
        &self,
        sorted_values: &mut SparseMatrix<T>,
        indices_to_sort_coumns: &mut SparseMatrix<ElementIndex>,
        sort_operator: &B,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn sorted_rows(&self, sort_operator: &B) -> Result<SparseMatrix<T>, SparseLinearAlgebraError>;

    fn sorted_columns(
        &self,
        sort_operator: &B,
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError>;

    fn indices_to_sort_rows(
        &self,
        sort_operator: &B,
    ) -> Result<SparseMatrix<ElementIndex>, SparseLinearAlgebraError>;

    fn indices_to_sort_columns(
        &self,
        sort_operator: &B,
    ) -> Result<SparseMatrix<ElementIndex>, SparseLinearAlgebraError>;
}

impl<T: ValueType, B: BinaryOperator<T, T, bool, T> + ReturnsBool> SortSparseMatrix<T, B>
    for SparseMatrix<T>
{
    fn sort_rows(&mut self, sort_operator: &B) -> Result<(), SparseLinearAlgebraError> {
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    self.graphblas_matrix(),
                    ptr::null_mut(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            unsafe { self.graphblas_matrix_ref() },
        )?;
        Ok(())
    }

    fn sort_columns(&mut self, sort_operator: &B) -> Result<(), SparseLinearAlgebraError> {
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    self.graphblas_matrix(),
                    ptr::null_mut(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    GrB_DESC_T0,
                )
            },
            unsafe { self.graphblas_matrix_ref() },
        )?;
        Ok(())
    }

    fn sorted_rows_and_indices(
        &self,
        sorted_values: &mut SparseMatrix<T>,
        indices_to_sort_rows: &mut SparseMatrix<ElementIndex>,
        sort_operator: &B,
    ) -> Result<(), SparseLinearAlgebraError> {
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    sorted_values.graphblas_matrix(),
                    indices_to_sort_rows.graphblas_matrix(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            unsafe { self.graphblas_matrix_ref() },
        )?;
        Ok(())
    }

    fn sorted_columns_and_indices(
        &self,
        sorted_values: &mut SparseMatrix<T>,
        indices_to_sort_columns: &mut SparseMatrix<ElementIndex>,
        sort_operator: &B,
    ) -> Result<(), SparseLinearAlgebraError> {
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    sorted_values.graphblas_matrix(),
                    indices_to_sort_columns.graphblas_matrix(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    GrB_DESC_T0,
                )
            },
            unsafe { self.graphblas_matrix_ref() },
        )?;
        Ok(())
    }

    fn sorted_rows(&self, sort_operator: &B) -> Result<SparseMatrix<T>, SparseLinearAlgebraError> {
        let sorted = SparseMatrix::<T>::new(self.context_ref(), &self.size()?)?;
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    sorted.graphblas_matrix(),
                    ptr::null_mut(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            unsafe { self.graphblas_matrix_ref() },
        )?;
        Ok(sorted)
    }

    fn sorted_columns(
        &self,
        sort_operator: &B,
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError> {
        let sorted = SparseMatrix::<T>::new(self.context_ref(), &self.size()?)?;
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    sorted.graphblas_matrix(),
                    ptr::null_mut(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    GrB_DESC_T0,
                )
            },
            unsafe { self.graphblas_matrix_ref() },
        )?;
        Ok(sorted)
    }

    fn indices_to_sort_rows(
        &self,
        sort_operator: &B,
    ) -> Result<SparseMatrix<ElementIndex>, SparseLinearAlgebraError> {
        let mut indices_to_sort_self =
            SparseMatrix::<ElementIndex>::new(self.context_ref(), &self.size()?)?;
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    ptr::null_mut(),
                    indices_to_sort_self.graphblas_matrix(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            unsafe { self.graphblas_matrix_ref() },
        )?;
        Ok(indices_to_sort_self)
    }

    fn indices_to_sort_columns(
        &self,
        sort_operator: &B,
    ) -> Result<SparseMatrix<ElementIndex>, SparseLinearAlgebraError> {
        let mut indices_to_sort_self =
            SparseMatrix::<ElementIndex>::new(self.context_ref(), &self.size()?)?;
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    ptr::null_mut(),
                    indices_to_sort_self.graphblas_matrix(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    GrB_DESC_T0,
                )
            },
            unsafe { self.graphblas_matrix_ref() },
        )?;
        Ok(indices_to_sort_self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList,
    };
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

        let element_list = MatrixElementList::<isize>::from_element_vector(vec![
            (1, 1, 2).into(),
            (2, 1, 1).into(),
            (2, 4, 3).into(),
            (4, 1, 6).into(),
            (6, 4, 4).into(),
        ]);

        let matrix = SparseMatrix::<isize>::from_element_list(
            &context.clone(),
            &(10, 10).into(),
            &element_list,
            &First::<isize, isize, isize, isize>::new(),
        )
        .unwrap();

        let mut sorted = SparseMatrix::new(&context, &matrix.size().unwrap()).unwrap();
        let mut indices = SparseMatrix::new(&context, &matrix.size().unwrap()).unwrap();

        let larger_than_operator = IsGreaterThan::<isize, isize, bool, isize>::new();

        matrix
            .sorted_columns_and_indices(&mut sorted, &mut indices, &larger_than_operator)
            .unwrap();

        assert_eq!(sorted.get_element_value(&(0, 1).into()).unwrap(), 6);
        assert_eq!(sorted.get_element_value(&(1, 1).into()).unwrap(), 2);
        assert_eq!(sorted.get_element_value(&(2, 1).into()).unwrap(), 1);
        assert_eq!(sorted.get_element_value(&(3, 1).into()).unwrap(), 0);

        assert_eq!(sorted.get_element_value(&(0, 4).into()).unwrap(), 4);
        assert_eq!(sorted.get_element_value(&(1, 4).into()).unwrap(), 3);

        assert_eq!(indices.get_element_value(&(0, 1).into()).unwrap(), 4);
        assert_eq!(indices.get_element_value(&(1, 1).into()).unwrap(), 1);
        assert_eq!(indices.get_element_value(&(2, 1).into()).unwrap(), 2);
        assert_eq!(indices.get_element_value(&(3, 1).into()).unwrap(), 0);

        assert_eq!(indices.get_element_value(&(0, 4).into()).unwrap(), 6);
        assert_eq!(indices.get_element_value(&(1, 4).into()).unwrap(), 2);
    }

}
