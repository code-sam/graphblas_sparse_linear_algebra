use crate::collections::sparse_matrix::operations::GetSparseMatrixSize;
use crate::collections::sparse_matrix::{GetGraphblasSparseMatrix, SparseMatrix};
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::graphblas_bindings::GxB_Matrix_subassign;
use crate::index::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::MatrixMask;
use crate::operators::options::GetOptionsForOperatorWithMatrixArguments;
use crate::value_type::ValueType;

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertMatrixIntoSubMatrixOperator {}
unsafe impl Sync for InsertMatrixIntoSubMatrixOperator {}

#[derive(Debug, Clone)]
pub struct InsertMatrixIntoSubMatrixOperator {}

impl InsertMatrixIntoSubMatrixOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait InsertMatrixIntoSubMatrix<MatrixToInsertInto>
where
    MatrixToInsertInto: ValueType,
{
    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
        mask_for_matrix_to_insert_into: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<MatrixToInsertInto: ValueType> InsertMatrixIntoSubMatrix<MatrixToInsertInto>
    for InsertMatrixIntoSubMatrixOperator
{
    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &impl GetGraphblasSparseMatrix,
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
        mask_for_matrix_to_insert_into: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArguments,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context_ref();

        let number_of_rows_to_insert_into = rows_to_insert_into
            .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
            .to_graphblas_index()?;

        let number_of_columns_to_insert_into = columns_to_insert_into
            .number_of_selected_elements(matrix_to_insert_into.column_width()?)?
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
                        GxB_Matrix_subassign(
                            GetGraphblasSparseMatrix::graphblas_matrix_ptr(matrix_to_insert_into),
                            mask_for_matrix_to_insert_into.graphblas_matrix_ptr(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix_ptr(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ptr_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GxB_Matrix_subassign(
                            GetGraphblasSparseMatrix::graphblas_matrix_ptr(matrix_to_insert_into),
                            mask_for_matrix_to_insert_into.graphblas_matrix_ptr(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix_ptr(),
                            row,
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ptr_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GxB_Matrix_subassign(
                            GetGraphblasSparseMatrix::graphblas_matrix_ptr(matrix_to_insert_into),
                            mask_for_matrix_to_insert_into.graphblas_matrix_ptr(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix_ptr(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ptr_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GxB_Matrix_subassign(
                            GetGraphblasSparseMatrix::graphblas_matrix_ptr(matrix_to_insert_into),
                            mask_for_matrix_to_insert_into.graphblas_matrix_ptr(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_insert.graphblas_matrix_ptr(),
                            row,
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ptr_ref() },
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
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First};

    use crate::collections::sparse_matrix::{MatrixElementList, Size};
    use crate::index::ElementIndex;
    use crate::operators::mask::SelectEntireMatrix;
    use crate::operators::options::OptionsForOperatorWithMatrixArguments;

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
            context.clone(),
            matrix_size,
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        let element_list_to_insert = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 2).into(),
            (1, 1, 3).into(),
            (0, 1, 11).into(),
            (1, 0, 12).into(),
        ]);

        let matrix_size_to_insert: Size = (2, 2).into();
        let matrix_to_insert = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            matrix_size_to_insert,
            element_list_to_insert,
            &First::<u8>::new(),
        )
        .unwrap();

        let mask_element_list = MatrixElementList::<bool>::from_element_vector(vec![
            (0, 0, true).into(),
            // (1, 1, true).into(),
            (1, 0, true).into(),
            (0, 1, true).into(),
        ]);
        let mask = SparseMatrix::<bool>::from_element_list(
            context.clone(),
            matrix_size_to_insert,
            mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let rows_to_insert: Vec<ElementIndex> = (0..2).collect();
        let rows_to_insert = ElementIndexSelector::Index(&rows_to_insert);
        let columns_to_insert: Vec<ElementIndex> = (0..2).collect();
        let columns_to_insert = ElementIndexSelector::Index(&columns_to_insert);

        let insert_operator = InsertMatrixIntoSubMatrixOperator::new();

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &matrix_to_insert,
                &Assignment::new(),
                &SelectEntireMatrix::new(context.clone()),
                &OptionsForOperatorWithMatrixArguments::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 7);
        assert_eq!(matrix.element_value_or_default(0, 0).unwrap(), 2);
        assert_eq!(matrix.element_value_or_default(1, 1).unwrap(), 3);
        assert_eq!(matrix.element_value_or_default(2, 2).unwrap(), 2);
        assert_eq!(matrix.element_value_or_default(2, 4).unwrap(), 10);
        assert_eq!(matrix.element_value_or_default(9, 5).unwrap(), 11);

        let mut matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            matrix_size,
            element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &matrix_to_insert,
                &Assignment::new(),
                &mask,
                &OptionsForOperatorWithMatrixArguments::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 7);
        assert_eq!(matrix.element_value_or_default(0, 0).unwrap(), 2);
        assert_eq!(matrix.element_value_or_default(2, 2).unwrap(), 2);
        assert_eq!(matrix.element_value_or_default(2, 4).unwrap(), 10);
        assert_eq!(matrix.element_value(2, 5).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(1, 1).unwrap(), 1);
    }
}
