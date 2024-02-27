use std::ptr;

use crate::collections::sparse_matrix::operations::GetSparseMatrixSize;
use crate::collections::sparse_matrix::{GetGraphblasSparseMatrix, SparseMatrix};
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::graphblas_bindings::{
    GxB_Matrix_subassign_BOOL, GxB_Matrix_subassign_FP32, GxB_Matrix_subassign_FP64,
    GxB_Matrix_subassign_INT16, GxB_Matrix_subassign_INT32, GxB_Matrix_subassign_INT64,
    GxB_Matrix_subassign_INT8, GxB_Matrix_subassign_UINT16, GxB_Matrix_subassign_UINT32,
    GxB_Matrix_subassign_UINT64, GxB_Matrix_subassign_UINT8,
};
use crate::index::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::{
    GetGraphblasDescriptor, GetMaskedOperatorWithMatrixArgumentOptions,
    GetMaskedOperatorWithTransposableArgumentsOptions, GetOperatorWithMatrixArgumentOptions,
};

use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_2_type_macro_for_all_value_types_and_typed_graphblas_function_with_scalar_type_conversion;
use crate::value_type::{ConvertScalar, ValueType};

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertScalarIntoSubMatrix {}
unsafe impl Sync for InsertScalarIntoSubMatrix {}

#[derive(Debug, Clone)]
pub struct InsertScalarIntoSubMatrix {}

impl InsertScalarIntoSubMatrix {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait InsertScalarIntoSubMatrixTrait<MatrixToInsertInto, ScalarToInsert>
where
    MatrixToInsertInto: ValueType,
    ScalarToInsert: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        scalar_to_insert: &ScalarToInsert,
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
        options: &impl GetOperatorWithMatrixArgumentOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        scalar_to_insert: &ScalarToInsert,
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
        mask_for_matrix_to_insert_into: &(impl GetGraphblasSparseMatrix + GetContext),
        options: &impl GetMaskedOperatorWithMatrixArgumentOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_insert_scalar_into_sub_matrix_trait {
    (
        $_value_type_matrix_to_insert_into:ty, $value_type_scalar_to_insert:ty, $graphblas_implemenation_type:ty, $graphblas_insert_function:ident, $convert_to_type:ident
    ) => {
        impl<MatrixToInsertInto: ValueType>
            InsertScalarIntoSubMatrixTrait<MatrixToInsertInto, $value_type_scalar_to_insert>
            for InsertScalarIntoSubMatrix
        {
            /// replace option applies to entire matrix_to_insert_to
            fn apply(
                &self,
                matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
                rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
                columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
                scalar_to_insert: &$value_type_scalar_to_insert,
                accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
                options: &impl GetOperatorWithMatrixArgumentOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = matrix_to_insert_into.context();
                let scalar_to_insert = scalar_to_insert.to_owned().to_type()?;

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
                                $graphblas_insert_function(
                                    matrix_to_insert_into.graphblas_matrix(),
                                    ptr::null_mut(),
                                    accumulator.accumulator_graphblas_type(),
                                    scalar_to_insert,
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
                                $graphblas_insert_function(
                                    matrix_to_insert_into.graphblas_matrix(),
                                    ptr::null_mut(),
                                    accumulator.accumulator_graphblas_type(),
                                    scalar_to_insert,
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
                                $graphblas_insert_function(
                                    matrix_to_insert_into.graphblas_matrix(),
                                    ptr::null_mut(),
                                    accumulator.accumulator_graphblas_type(),
                                    scalar_to_insert,
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
                                $graphblas_insert_function(
                                    matrix_to_insert_into.graphblas_matrix(),
                                    ptr::null_mut(),
                                    accumulator.accumulator_graphblas_type(),
                                    scalar_to_insert,
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
                matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
                rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
                columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
                scalar_to_insert: &$value_type_scalar_to_insert,
                accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
                mask_for_matrix_to_insert_into: &(impl GetGraphblasSparseMatrix + GetContext),
                options: &impl GetMaskedOperatorWithMatrixArgumentOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = matrix_to_insert_into.context();
                let scalar_to_insert = scalar_to_insert.to_owned().to_type()?;

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
                                $graphblas_insert_function(
                                    matrix_to_insert_into.graphblas_matrix(),
                                    mask_for_matrix_to_insert_into.graphblas_matrix(),
                                    accumulator.accumulator_graphblas_type(),
                                    scalar_to_insert,
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
                                $graphblas_insert_function(
                                    matrix_to_insert_into.graphblas_matrix(),
                                    mask_for_matrix_to_insert_into.graphblas_matrix(),
                                    accumulator.accumulator_graphblas_type(),
                                    scalar_to_insert,
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
                                $graphblas_insert_function(
                                    matrix_to_insert_into.graphblas_matrix(),
                                    mask_for_matrix_to_insert_into.graphblas_matrix(),
                                    accumulator.accumulator_graphblas_type(),
                                    scalar_to_insert,
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
                                $graphblas_insert_function(
                                    matrix_to_insert_into.graphblas_matrix(),
                                    mask_for_matrix_to_insert_into.graphblas_matrix(),
                                    accumulator.accumulator_graphblas_type(),
                                    scalar_to_insert,
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
    };
}

implement_2_type_macro_for_all_value_types_and_typed_graphblas_function_with_scalar_type_conversion!(
    implement_insert_scalar_into_sub_matrix_trait,
    GxB_Matrix_subassign
);

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
    use crate::operators::options::{
        GetMaskedOperatorWithMatrixArgumentOptions, MaskedOperatorWithMatrixArgumentOptions,
        MaskedOperatorWithTransposableArgumentsOptions, OperatorOptions,
        OperatorWithMatrixArgumentOptions,
    };

    #[test]
    fn test_insert_scalar_into_matrix() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            // (2, 5, 11).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
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
            &(3, 6).into(),
            &mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let rows_to_insert: Vec<ElementIndex> = (0..3).collect();
        let rows_to_insert = ElementIndexSelector::Index(&rows_to_insert);
        let columns_to_insert: Vec<ElementIndex> = (0..6).collect();
        let columns_to_insert = ElementIndexSelector::Index(&columns_to_insert);

        let insert_operator = InsertScalarIntoSubMatrix::new();

        let scalar_to_insert: u8 = 8;

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &scalar_to_insert,
                &Assignment::new(),
                &OperatorWithMatrixArgumentOptions::new_default(),
            )
            .unwrap();

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 18);
        assert_eq!(matrix.element_value_or_default(&0, &0).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&2, &4).unwrap(), 8);
        assert_eq!(matrix.element_value(&9, &14).unwrap(), None);

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
                &scalar_to_insert,
                &Assignment::new(),
                &mask,
                &MaskedOperatorWithMatrixArgumentOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&2, &4).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&2, &5).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&1, &1).unwrap(), 1);
    }
}
