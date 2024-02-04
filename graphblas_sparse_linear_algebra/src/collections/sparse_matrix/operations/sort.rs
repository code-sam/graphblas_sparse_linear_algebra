use std::ptr;

use once_cell::sync::Lazy;

use suitesparse_graphblas_sys::{GrB_DESC_T0, GxB_Matrix_sort};

use crate::context::{CallGraphBlasContext, GetContext};
use crate::index::ElementIndex;
use crate::operators::options::{GetGraphblasDescriptor, MutateOperatorOptions, OperatorOptions};
use crate::{
    collections::sparse_matrix::{GetGraphblasSparseMatrix, SparseMatrix},
    error::SparseLinearAlgebraError,
    operators::binary_operator::{BinaryOperator, ReturnsBool},
    value_type::ValueType,
};

use super::GetSparseMatrixSize;

static DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS: Lazy<OperatorOptions> =
    Lazy::new(|| OperatorOptions::new_default());

// REVIEW: support typecasting for indices and the evaluation domain of the binary operator
pub trait SortSparseMatrix<T: ValueType, B: BinaryOperator<T>> {
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

impl<T: ValueType, B: BinaryOperator<T> + ReturnsBool> SortSparseMatrix<T, B> for SparseMatrix<T> {
    fn sort_rows(&mut self, sort_operator: &B) -> Result<(), SparseLinearAlgebraError> {
        self.context_ref().call(
            || unsafe {
                GxB_Matrix_sort(
                    self.graphblas_matrix(),
                    ptr::null_mut(),
                    sort_operator.graphblas_type(),
                    self.graphblas_matrix(),
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
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
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
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
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
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
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
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

// TODO: review how safe these untyped functions would be, and if such untyped function would be desireable.
// pub fn sort_sparse_matrix_rows<T: ValueType>(
//     matrix: &mut (impl GetGraphblasSparseMatrix + GetContext),
//     sort_operator: &(impl BinaryOperator<T> + ReturnsBool),
// ) -> Result<(), SparseLinearAlgebraError> {
//     matrix.context_ref().call(
//         || unsafe {
//             GxB_Matrix_sort(
//                 matrix.graphblas_matrix(),
//                 ptr::null_mut(),
//                 sort_operator.graphblas_type(),
//                 matrix.graphblas_matrix(),
//                 DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
//             )
//         },
//         unsafe { matrix.graphblas_matrix_ref() },
//     )?;
//     Ok(())
// }

// fn sort_sparse_matrix_columns<T: ValueType>(
//     matrix: &mut (impl GetGraphblasSparseMatrix + GetContext),
//     sort_operator: &(impl BinaryOperator<T> + ReturnsBool),
// ) -> Result<(), SparseLinearAlgebraError> {
//     matrix.context_ref().call(
//         || unsafe {
//             GxB_Matrix_sort(
//                 matrix.graphblas_matrix(),
//                 ptr::null_mut(),
//                 sort_operator.graphblas_type(),
//                 matrix.graphblas_matrix(),
//                 GrB_DESC_T0,
//             )
//         },
//         unsafe { matrix.graphblas_matrix_ref() },
//     )?;
//     Ok(())
// }

// fn sorted_rows_and_indices<T: ValueType>(
//     matrix: &(impl GetGraphblasSparseMatrix + GetContext),
//     sorted_values: &mut SparseMatrix<T>,
//     indices_to_sort_rows: &mut SparseMatrix<ElementIndex>,
//     sort_operator: &(impl BinaryOperator<T> + ReturnsBool),
// ) -> Result<(), SparseLinearAlgebraError> {
//     matrix.context_ref().call(
//         || unsafe {
//             GxB_Matrix_sort(
//                 sorted_values.graphblas_matrix(),
//                 indices_to_sort_rows.graphblas_matrix(),
//                 sort_operator.graphblas_type(),
//                 matrix.graphblas_matrix(),
//                 DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
//             )
//         },
//         unsafe { matrix.graphblas_matrix_ref() },
//     )?;
//     Ok(())
// }

// fn sorted_columns_and_indices<T: ValueType>(
//     &self,
//     sorted_values: &mut SparseMatrix<T>,
//     indices_to_sort_columns: &mut SparseMatrix<ElementIndex>,
//     sort_operator: &B,
// ) -> Result<(), SparseLinearAlgebraError> {
//     self.context_ref().call(
//         || unsafe {
//             GxB_Matrix_sort(
//                 sorted_values.graphblas_matrix(),
//                 indices_to_sort_columns.graphblas_matrix(),
//                 sort_operator.graphblas_type(),
//                 self.graphblas_matrix(),
//                 GrB_DESC_T0,
//             )
//         },
//         unsafe { self.graphblas_matrix_ref() },
//     )?;
//     Ok(())
// }

// fn sorted_rows<T: ValueType>(
//     &self,
//     sort_operator: &B,
// ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError> {
//     let sorted = SparseMatrix::<T>::new(self.context_ref(), &self.size()?)?;
//     self.context_ref().call(
//         || unsafe {
//             GxB_Matrix_sort(
//                 sorted.graphblas_matrix(),
//                 ptr::null_mut(),
//                 sort_operator.graphblas_type(),
//                 self.graphblas_matrix(),
//                 DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
//             )
//         },
//         unsafe { self.graphblas_matrix_ref() },
//     )?;
//     Ok(sorted)
// }

// fn sorted_columns<T: ValueType>(
//     &self,
//     sort_operator: &B,
// ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError> {
//     let sorted = SparseMatrix::<T>::new(self.context_ref(), &self.size()?)?;
//     self.context_ref().call(
//         || unsafe {
//             GxB_Matrix_sort(
//                 sorted.graphblas_matrix(),
//                 ptr::null_mut(),
//                 sort_operator.graphblas_type(),
//                 self.graphblas_matrix(),
//                 GrB_DESC_T0,
//             )
//         },
//         unsafe { self.graphblas_matrix_ref() },
//     )?;
//     Ok(sorted)
// }

// fn indices_to_sort_rows<T: ValueType>(
//     &self,
//     sort_operator: &B,
// ) -> Result<SparseMatrix<ElementIndex>, SparseLinearAlgebraError> {
//     let mut indices_to_sort_self =
//         SparseMatrix::<ElementIndex>::new(self.context_ref(), &self.size()?)?;
//     self.context_ref().call(
//         || unsafe {
//             GxB_Matrix_sort(
//                 ptr::null_mut(),
//                 indices_to_sort_self.graphblas_matrix(),
//                 sort_operator.graphblas_type(),
//                 self.graphblas_matrix(),
//                 DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.graphblas_descriptor(),
//             )
//         },
//         unsafe { self.graphblas_matrix_ref() },
//     )?;
//     Ok(indices_to_sort_self)
// }

// fn indices_to_sort_columns<T: ValueType>(
//     &self,
//     sort_operator: &B,
// ) -> Result<SparseMatrix<ElementIndex>, SparseLinearAlgebraError> {
//     let mut indices_to_sort_self =
//         SparseMatrix::<ElementIndex>::new(self.context_ref(), &self.size()?)?;
//     self.context_ref().call(
//         || unsafe {
//             GxB_Matrix_sort(
//                 ptr::null_mut(),
//                 indices_to_sort_self.graphblas_matrix(),
//                 sort_operator.graphblas_type(),
//                 self.graphblas_matrix(),
//                 GrB_DESC_T0,
//             )
//         },
//         unsafe { self.graphblas_matrix_ref() },
//     )?;
//     Ok(indices_to_sort_self)
// }

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::get_element_value::GetSparseMatrixElementValue;
    use crate::collections::sparse_matrix::operations::FromMatrixElementList;
    use crate::collections::sparse_matrix::MatrixElementList;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{First, IsGreaterThan};

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
            &context.to_owned(),
            &(10, 10).into(),
            &element_list,
            &First::<isize>::new(),
        )
        .unwrap();

        let mut sorted = SparseMatrix::new(&context, &matrix.size().unwrap()).unwrap();
        let mut indices = SparseMatrix::new(&context, &matrix.size().unwrap()).unwrap();

        let larger_than_operator = IsGreaterThan::<isize>::new();

        matrix
            .sorted_columns_and_indices(&mut sorted, &mut indices, &larger_than_operator)
            .unwrap();

        assert_eq!(sorted.element_value(&0, &1).unwrap(), Some(6));
        assert_eq!(sorted.element_value(&1, &1).unwrap(), Some(2));
        assert_eq!(sorted.element_value(&2, &1).unwrap(), Some(1));
        assert_eq!(sorted.element_value(&3, &1).unwrap(), None);

        assert_eq!(sorted.element_value(&0, &4).unwrap(), Some(4));
        assert_eq!(sorted.element_value(&1, &4).unwrap(), Some(3));

        assert_eq!(indices.element_value(&0, &1).unwrap(), Some(4));
        assert_eq!(indices.element_value(&1, &1).unwrap(), Some(1));
        assert_eq!(indices.element_value(&2, &1).unwrap(), Some(2));
        assert_eq!(indices.element_value(&3, &1).unwrap(), None);

        assert_eq!(indices.element_value(&0, &4).unwrap(), Some(6));
        assert_eq!(indices.element_value(&1, &4).unwrap(), Some(2));
    }
}
