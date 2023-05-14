use std::marker::PhantomData;
use std::ptr;

use crate::collections::sparse_matrix::{
    GraphblasSparseMatrixTrait, SparseMatrix, SparseMatrixTrait,
};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::index::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::OperatorOptions;
use crate::value_type::ValueType;

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_assign,
};

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<MatrixToInsertInto: ValueType> Send for InsertMatrixIntoMatrix<MatrixToInsertInto> {}
unsafe impl<MatrixToInsertInto: ValueType> Sync for InsertMatrixIntoMatrix<MatrixToInsertInto> {}

#[derive(Debug, Clone)]
pub struct InsertMatrixIntoMatrix<MatrixToInsertInto: ValueType> {
    _matrix_to_insert_into: PhantomData<MatrixToInsertInto>,

    accumulator: GrB_BinaryOp,
    options: GrB_Descriptor,
}

impl<MatrixToInsertInto> InsertMatrixIntoMatrix<MatrixToInsertInto>
where
    MatrixToInsertInto: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
    ) -> Self {
        Self {
            accumulator: accumulator.accumulator_graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _matrix_to_insert_into: PhantomData,
        }
    }
}

pub trait InsertMatrixIntoMatrixTrait<MatrixToInsertInto>
where
    MatrixToInsertInto: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &(impl GraphblasSparseMatrixTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        mask_for_matrix_to_insert_into: &(impl GraphblasSparseMatrixTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<MatrixToInsertInto: ValueType> InsertMatrixIntoMatrixTrait<MatrixToInsertInto>
    for InsertMatrixIntoMatrix<MatrixToInsertInto>
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &(impl GraphblasSparseMatrixTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

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
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            matrix_to_insert.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            self.options,
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
                            self.accumulator,
                            matrix_to_insert.graphblas_matrix(),
                            row,
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            self.options,
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
                            self.accumulator,
                            matrix_to_insert.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            self.options,
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
                            self.accumulator,
                            matrix_to_insert.graphblas_matrix(),
                            row,
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            self.options,
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
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        mask_for_matrix_to_insert_into: &(impl GraphblasSparseMatrixTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

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
                        GrB_Matrix_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_matrix_to_insert_into.graphblas_matrix(),
                            self.accumulator,
                            matrix_to_insert.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            self.options,
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
                            self.accumulator,
                            matrix_to_insert.graphblas_matrix(),
                            row,
                            number_of_rows_to_insert_into,
                            column.as_ptr(),
                            number_of_columns_to_insert_into,
                            self.options,
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
                            self.accumulator,
                            matrix_to_insert.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            self.options,
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
                            self.accumulator,
                            matrix_to_insert.graphblas_matrix(),
                            row,
                            number_of_rows_to_insert_into,
                            column,
                            number_of_columns_to_insert_into,
                            self.options,
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

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::index::ElementIndex;
    use crate::operators::binary_operator::{Assignment, First};

    #[test]
    fn test_insert_matrix_into_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

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

        let insert_operator =
            InsertMatrixIntoMatrix::new(&OperatorOptions::new_default(), &Assignment::<u8>::new());

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &matrix_to_insert,
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 5);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), None);
        assert_eq!(
            matrix.get_element_value_or_default(&(1, 1).into()).unwrap(),
            2
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(2, 2).into()).unwrap(),
            3
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(2, 4).into()).unwrap(),
            11
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(9, 5).into()).unwrap(),
            11
        );

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
                &mask,
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 5);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), None);
        assert_eq!(
            matrix.get_element_value_or_default(&(2, 2).into()).unwrap(),
            3
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(2, 4).into()).unwrap(),
            11
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(2, 5).into()).unwrap(),
            12
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(1, 1).into()).unwrap(),
            1
        );
    }
}
