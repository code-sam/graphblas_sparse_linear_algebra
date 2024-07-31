use crate::collections::sparse_matrix::operations::sparse_matrix_column_width;
use crate::collections::sparse_matrix::operations::sparse_matrix_row_height;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::CallGraphBlasContext;
use crate::error::SparseLinearAlgebraError;
use crate::index::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::operators::binary_operator::AccumulatorBinaryOperator;

use crate::operators::mask::MatrixMask;
use crate::operators::options::GetOptionsForOperatorWithMatrixArgument;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_2_type_macro_for_all_value_types_and_typed_graphblas_function_with_scalar_type_conversion;
use crate::value_type::{ConvertScalar, ValueType};

use crate::graphblas_bindings::{
    GrB_Matrix_assign_BOOL, GrB_Matrix_assign_FP32, GrB_Matrix_assign_FP64,
    GrB_Matrix_assign_INT16, GrB_Matrix_assign_INT32, GrB_Matrix_assign_INT64,
    GrB_Matrix_assign_INT8, GrB_Matrix_assign_UINT16, GrB_Matrix_assign_UINT32,
    GrB_Matrix_assign_UINT64, GrB_Matrix_assign_UINT8,
};

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertScalarIntoMatrixOperator {}
unsafe impl Sync for InsertScalarIntoMatrixOperator {}

#[derive(Debug, Clone)]
pub struct InsertScalarIntoMatrixOperator {}

impl InsertScalarIntoMatrixOperator {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait InsertScalarIntoMatrix<AccumulatorEvaluationDomain, ScalarToInsert>
where
    AccumulatorEvaluationDomain: ValueType,
    ScalarToInsert: ValueType,
{
    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut impl GetGraphblasSparseMatrix,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        scalar_to_insert: ScalarToInsert,
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        mask_for_matrix_to_insert_into: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArgument,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_insert_scalar_into_matrix_trait {
    (
        $value_type_matrix_to_insert_into:ty, $value_type_scalar_to_insert:ty, $graphblas_implementation_type:ty, $graphblas_insert_function:ident, $convert_to_type:ident
    ) => {
        impl<AccumulatorEvaluationDomain: ValueType>
            InsertScalarIntoMatrix<AccumulatorEvaluationDomain, $value_type_scalar_to_insert>
            for InsertScalarIntoMatrixOperator
        {
            /// mask and replace option apply to entire matrix_to_insert_to
            fn apply(
                &self,
                matrix_to_insert_into: &mut impl GetGraphblasSparseMatrix,
                rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
                columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
                scalar_to_insert: $value_type_scalar_to_insert,
                accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
                mask_for_matrix_to_insert_into: &impl MatrixMask,
                options: &impl GetOptionsForOperatorWithMatrixArgument,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = matrix_to_insert_into.context_ref();
                let scalar_to_insert = scalar_to_insert.to_type()?;

                let number_of_rows_to_insert_into = rows_to_insert_into
                    .number_of_selected_elements(sparse_matrix_row_height(matrix_to_insert_into)?)?
                    .to_graphblas_index()?;

                let number_of_columns_to_insert_into = columns_to_insert_into
                    .number_of_selected_elements(sparse_matrix_column_width(
                        matrix_to_insert_into,
                    )?)?
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
    implement_insert_scalar_into_matrix_trait,
    GrB_Matrix_assign
);

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
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::SelectEntireMatrix;
    use crate::operators::options::OptionsForOperatorWithMatrixArgument;

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
            context.clone(),
            matrix_size,
            element_list.clone(),
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
            context.clone(),
            matrix_size,
            mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let rows_to_insert: Vec<ElementIndex> = (0..3).collect();
        let rows_to_insert = ElementIndexSelector::Index(&rows_to_insert);
        let columns_to_insert: Vec<ElementIndex> = (0..6).collect();
        let columns_to_insert = ElementIndexSelector::Index(&columns_to_insert);

        let insert_operator = InsertScalarIntoMatrixOperator::new();

        let scalar_to_insert: u8 = 8;

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                scalar_to_insert,
                &Assignment::<u8>::new(),
                &SelectEntireMatrix::new(context.clone()),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 18);
        assert_eq!(matrix.element_value_or_default(&0, &0).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&2, &4).unwrap(), 8);
        assert_eq!(matrix.element_value(&9, &14).unwrap(), None);

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
                scalar_to_insert,
                &Assignment::<u8>::new(),
                &mask,
                &OptionsForOperatorWithMatrixArgument::new_default(),
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

    #[test]
    fn test_insert_scalar_into_matrix_with_type_casting() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            // (2, 5, 11).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let mut matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            matrix_size,
            element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let rows_to_insert: Vec<ElementIndex> = (0..3).collect();
        let rows_to_insert = ElementIndexSelector::Index(&rows_to_insert);
        let columns_to_insert: Vec<ElementIndex> = (0..6).collect();
        let columns_to_insert = ElementIndexSelector::Index(&columns_to_insert);

        let insert_operator = InsertScalarIntoMatrixOperator::new();

        let scalar_to_insert: f32 = 8.0;

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                scalar_to_insert,
                &Assignment::<u8>::new(),
                &SelectEntireMatrix::new(context.clone()),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", matrix);
        assert_eq!(matrix.number_of_stored_elements().unwrap(), 18);
        assert_eq!(matrix.element_value_or_default(&0, &0).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 8);
        assert_eq!(matrix.element_value_or_default(&2, &4).unwrap(), 8);
        assert_eq!(matrix.element_value(&9, &14).unwrap(), None);
    }
}
