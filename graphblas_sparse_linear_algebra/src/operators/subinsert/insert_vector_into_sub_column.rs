use std::ptr;

use crate::collections::sparse_matrix::operations::GetSparseMatrixSize;
use crate::collections::sparse_matrix::{GetGraphblasSparseMatrix, SparseMatrix};
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::graphblas_bindings::GxB_Col_subassign;
use crate::index::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::options::{
    GetGraphblasDescriptor, GetMaskedOperatorWithMatrixArgumentOptions,
    GetOperatorWithMatrixArgumentOptions,
};

use crate::value_type::ValueType;

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertVectorIntoSubColumn {}
unsafe impl Sync for InsertVectorIntoSubColumn {}

#[derive(Debug, Clone)]
pub struct InsertVectorIntoSubColumn {}

impl InsertVectorIntoSubColumn {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait InsertVectorIntoSubColumnTrait<MatrixToInsertInto>
where
    MatrixToInsertInto: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        column_indices_to_insert_into: &ElementIndexSelector,
        column_to_insert_into: &ElementIndex,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
        options: &impl GetOperatorWithMatrixArgumentOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        column_indices_to_insert_into: &ElementIndexSelector,
        column_to_insert_into: &ElementIndex,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        mask_for_vector_to_insert_into: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
        options: &impl GetMaskedOperatorWithMatrixArgumentOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<MatrixToInsertInto: ValueType> InsertVectorIntoSubColumnTrait<MatrixToInsertInto>
    for InsertVectorIntoSubColumn
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        column_indices_to_insert_into: &ElementIndexSelector,
        column_to_insert_into: &ElementIndex,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
        options: &impl GetOperatorWithMatrixArgumentOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

        let number_of_indices_to_insert_into = column_indices_to_insert_into
            .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
            .to_graphblas_index()?;

        let indices_to_insert_into = column_indices_to_insert_into.to_graphblas_type()?;
        let column_to_insert_into = column_to_insert_into.to_graphblas_index()?;

        match indices_to_insert_into {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GxB_Col_subassign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            column_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }

            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GxB_Col_subassign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            index,
                            number_of_indices_to_insert_into,
                            column_to_insert_into,
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
        column_indices_to_insert_into: &ElementIndexSelector,
        column_to_insert_into: &ElementIndex,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        mask_for_column_to_insert_into: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
        options: &impl GetMaskedOperatorWithMatrixArgumentOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

        let number_of_indices_to_insert_into = column_indices_to_insert_into
            .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
            .to_graphblas_index()?;

        let indices_to_insert_into = column_indices_to_insert_into.to_graphblas_type()?;
        let column_to_insert_into = column_to_insert_into.to_graphblas_index()?;

        match indices_to_insert_into {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GxB_Col_subassign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_column_to_insert_into.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            column_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }

            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GxB_Col_subassign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_column_to_insert_into.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            index,
                            number_of_indices_to_insert_into,
                            column_to_insert_into,
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
    use crate::collections::sparse_vector::operations::FromVectorElementList;
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::index::ElementIndex;
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::options::{
        MaskedOperatorWithMatrixArgumentOptions, OperatorWithMatrixArgumentOptions,
    };

    #[test]
    fn test_insert_vector_into_column() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 2, 1).into(),
            (2, 2, 2).into(),
            (4, 2, 10).into(),
            (5, 2, 12).into(),
        ]);

        let matrix_size: Size = (10, 5).into();
        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let element_list_to_insert = VectorElementList::<u8>::from_element_vector(vec![
            (1, 2).into(),
            (2, 3).into(),
            (4, 11).into(),
            // (5, 11).into(),
        ]);

        let vector_to_insert_length: usize = 5;
        let vector_to_insert = SparseVector::<u8>::from_element_list(
            &context,
            &vector_to_insert_length,
            &element_list_to_insert,
            &First::<u8>::new(),
        )
        .unwrap();

        let mask_element_list = VectorElementList::<bool>::from_element_vector(vec![
            // (1, 1, true).into(),
            (2, true).into(),
            (4, true).into(),
            // (5, true).into(),
        ]);
        let mask = SparseVector::<bool>::from_element_list(
            &context,
            &vector_to_insert_length,
            &mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let indices_to_insert: Vec<ElementIndex> = (0..vector_to_insert_length).collect();
        let indices_to_insert = ElementIndexSelector::Index(&indices_to_insert);

        let insert_operator = InsertVectorIntoSubColumn::new();

        let column_to_insert_into: ElementIndex = 2;

        insert_operator
            .apply(
                &mut matrix,
                &indices_to_insert,
                &column_to_insert_into,
                &vector_to_insert,
                &Assignment::new(),
                &OperatorWithMatrixArgumentOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&1, &2).unwrap(), 2);
        assert_eq!(matrix.element_value_or_default(&5, &2).unwrap(), 12);

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
                &indices_to_insert,
                &column_to_insert_into,
                &vector_to_insert,
                &mask,
                &Assignment::new(),
                &MaskedOperatorWithMatrixArgumentOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 3);
        assert_eq!(matrix.element_value_or_default(&1, &2).unwrap(), 1);
        assert_eq!(matrix.element_value_or_default(&5, &2).unwrap(), 12);
    }
}
