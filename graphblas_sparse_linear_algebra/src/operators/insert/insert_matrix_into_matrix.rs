use std::ptr;

use crate::collections::sparse_matrix::operations::sparse_matrix_column_width;
use crate::collections::sparse_matrix::operations::sparse_matrix_row_height;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::index::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::GetGraphblasDescriptor;

use crate::value_type::ValueType;

use crate::graphblas_bindings::GrB_Matrix_assign;

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertMatrixIntoMatrix {}
unsafe impl Sync for InsertMatrixIntoMatrix {}

#[derive(Debug, Clone)]
pub struct InsertMatrixIntoMatrix {}

impl InsertMatrixIntoMatrix {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait InsertMatrixIntoMatrixTrait<AccumulatorEvaluationDomain>
where
    AccumulatorEvaluationDomain: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut (impl GetGraphblasSparseMatrix + GetContext),
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask(
        &self,
        matrix_to_insert_into: &mut (impl GetGraphblasSparseMatrix + GetContext),
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        mask_for_matrix_to_insert_into: &(impl GetGraphblasSparseMatrix + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<AccumulatorEvaluationDomain: ValueType>
    InsertMatrixIntoMatrixTrait<AccumulatorEvaluationDomain> for InsertMatrixIntoMatrix
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut (impl GetGraphblasSparseMatrix + GetContext),
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

        let number_of_rows_to_insert_into = rows_to_insert_into
            .number_of_selected_elements(sparse_matrix_row_height(matrix_to_insert_into)?)?
            .to_graphblas_index()?;

        let number_of_columns_to_insert_into = columns_to_insert_into
            .number_of_selected_elements(sparse_matrix_column_width(matrix_to_insert_into)?)?
            .to_graphblas_index()?;

        let rows_to_insert_into = rows_to_insert_into.to_graphblas_type()?;
        let columns_to_insert_into = columns_to_insert_into.to_graphblas_type()?;

        match (rows_to_insert_into, columns_to_insert_into) {
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix(),
                            row,
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix(),
                            row,
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }
        }

        Ok(())
    }

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask(
        &self,
        matrix_to_insert_into: &mut (impl GetGraphblasSparseMatrix + GetContext),
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        mask_for_matrix_to_insert_into: &(impl GetGraphblasSparseMatrix + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

        let number_of_rows_to_insert_into = rows_to_insert_into
            .number_of_selected_elements(sparse_matrix_row_height(matrix_to_insert_into)?)?
            .to_graphblas_index()?;

        let number_of_columns_to_insert_into = columns_to_insert_into
            .number_of_selected_elements(sparse_matrix_column_width(matrix_to_insert_into)?)?
            .to_graphblas_index()?;

        let rows_to_insert_into = rows_to_insert_into.to_graphblas_type()?;
        let columns_to_insert_into = columns_to_insert_into.to_graphblas_type()?;

        match (rows_to_insert_into, columns_to_insert_into) {
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_matrix_to_insert_into.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_matrix_to_insert_into.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix(),
                            row,
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_matrix_to_insert_into.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_matrix_to_insert_into.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix(),
                            row,
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementValue,
    };
    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::index::ElementIndex;
    use crate::operators::binary_operator::{Assignment, First, Plus};
    use crate::operators::options::OperatorOptions;

    #[test]
    fn test_insert_matrix_into_matrix() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (9, 5, 11).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let element_list_to_insert = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 2).into(),
            (2, 2, 3).into(),
            (2, 4, 11).into(),
            (2, 5, 12).into(),
        ]);

        let matrix_size_to_insert: Size = (5, 10).into();
        let matrix_to_insert = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size_to_insert,
            &element_list_to_insert,
            &First::<u8>::new(),
        )
        .unwrap();

        let mask_element_list = MatrixElementList::<bool>::from_element_vector(vec![
            // (1, 1, true).into(),
            (2, 2, true).into(),
            (2, 4, true).into(),
            (2, 5, true).into(),
        ]);
        let mask = SparseMatrix::<bool>::from_element_list(
            &context,
            &matrix_size,
            &mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let rows_to_insert: Vec<ElementIndex> = (0..5).collect();
        let rows_to_insert = ElementIndexSelector::Index(&rows_to_insert);
        let columns_to_insert: Vec<ElementIndex> = (0..10).collect();
        let columns_to_insert = ElementIndexSelector::Index(&columns_to_insert);

        let insert_operator = InsertMatrixIntoMatrix::new();

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &matrix_to_insert,
                &Assignment::<u8>::new(),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 5);
        assert_eq!(matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&1, &1).unwrap(), 2);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 3);
        assert_eq!(matrix.element_value_or_default(&2, &4).unwrap(), 11);
        assert_eq!(matrix.element_value_or_default(&9, &5).unwrap(), 11);

        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply_with_mask(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &matrix_to_insert,
                &Assignment::<u8>::new(),
                &mask,
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 5);
        assert_eq!(matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 3);
        assert_eq!(matrix.element_value_or_default(&2, &4).unwrap(), 11);
        assert_eq!(matrix.element_value_or_default(&2, &5).unwrap(), 12);
        assert_eq!(matrix.element_value_or_default(&1, &1).unwrap(), 1);
    }

    #[test]
    fn test_insert_matrix_into_matrix_with_other_typed_accumulator() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (9, 5, 11).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let insert_operator = InsertMatrixIntoMatrix::new();

        let matrix_to_insert = matrix.clone();

        insert_operator
            .apply(
                &mut matrix,
                &ElementIndexSelector::All,
                &ElementIndexSelector::All,
                &matrix_to_insert,
                &Plus::<f32>::new(),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&1, &1).unwrap(), 2);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 4);
    }
}
