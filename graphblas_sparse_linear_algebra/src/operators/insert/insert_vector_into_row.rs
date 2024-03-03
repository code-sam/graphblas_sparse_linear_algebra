use crate::collections::sparse_matrix::operations::sparse_matrix_row_height;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::index::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::GetOptionsForOperatorWithMatrixArgument;

use crate::value_type::ValueType;

use crate::graphblas_bindings::GrB_Row_assign;

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertVectorIntoRow {}
unsafe impl Sync for InsertVectorIntoRow {}

#[derive(Debug, Clone)]
pub struct InsertVectorIntoRow {}

impl InsertVectorIntoRow {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait InsertVectorIntoRowTrait<AccumulatorEvaluationDomain>
where
    AccumulatorEvaluationDomain: ValueType,
{
    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut (impl GetGraphblasSparseMatrix + GetContext),
        row_indices_to_insert_into: &ElementIndexSelector,
        row_to_insert_into: &ElementIndex,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        mask_for_row_to_insert_into: &(impl VectorMask + GetContext),
        options: &impl GetOptionsForOperatorWithMatrixArgument,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<AccumulatorEvaluationDomain: ValueType> InsertVectorIntoRowTrait<AccumulatorEvaluationDomain>
    for InsertVectorIntoRow
{
    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut (impl GetGraphblasSparseMatrix + GetContext),
        row_indices_to_insert_into: &ElementIndexSelector,
        row_to_insert_into: &ElementIndex,
        vector_to_insert: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<AccumulatorEvaluationDomain>,
        mask_for_row_to_insert_into: &(impl VectorMask + GetContext),
        options: &impl GetOptionsForOperatorWithMatrixArgument,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

        let number_of_indices_to_insert_into = row_indices_to_insert_into
            .number_of_selected_elements(sparse_matrix_row_height(matrix_to_insert_into)?)?
            .to_graphblas_index()?;

        let indices_to_insert_into = row_indices_to_insert_into.to_graphblas_type()?;
        let row_to_insert_into = row_to_insert_into.to_graphblas_index()?;

        match indices_to_insert_into {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GrB_Row_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_row_to_insert_into.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            row_to_insert_into,
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }

            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GrB_Row_assign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_row_to_insert_into.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            vector_to_insert.graphblas_vector(),
                            row_to_insert_into,
                            index,
                            number_of_indices_to_insert_into,
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
    use crate::collections::sparse_vector::operations::FromVectorElementList;
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First};

    use crate::collections::sparse_matrix::{
        GetMatrixDimensions, MatrixElementList, Size, SparseMatrix,
    };
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::options::OptionsForOperatorWithMatrixArgument;

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
            (1, false).into(),
            (2, true).into(),
            (4, true).into(),
            // (5, true).into(),
        ]);
        let mask = SparseVector::<bool>::from_element_list(
            &context,
            matrix_size.column_width_ref(),
            &mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let indices_to_insert: Vec<ElementIndex> = (0..vector_to_insert_length).collect();
        let indices_to_insert = ElementIndexSelector::Index(&indices_to_insert);

        let insert_operator = InsertVectorIntoRow::new();

        let row_to_insert_into: ElementIndex = 2;

        insert_operator
            .apply(
                &mut matrix,
                &indices_to_insert,
                &row_to_insert_into,
                &vector_to_insert,
                &Assignment::<u8>::new(),
                &SelectEntireVector::new(&context),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 6);
        assert_eq!(matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&1, &2).unwrap(), 1);
        assert_eq!(matrix.element_value_or_default(&5, &2).unwrap(), 12);
        assert_eq!(matrix.element_value(&2, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&2, &4).unwrap(), 11);

        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply(
                &mut matrix,
                &indices_to_insert,
                &row_to_insert_into,
                &vector_to_insert,
                &Assignment::<u8>::new(),
                &mask,
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 5);
        assert_eq!(matrix.element_value(&0, &0).unwrap(), None);
        assert_eq!(matrix.element_value_or_default(&2, &2).unwrap(), 3);
        assert_eq!(matrix.element_value_or_default(&1, &2).unwrap(), 1);
        assert_eq!(matrix.element_value_or_default(&5, &2).unwrap(), 12);
        assert_eq!(matrix.element_value_or_default(&2, &4).unwrap(), 11);
        assert_eq!(matrix.element_value(&2, &1).unwrap(), None);
    }
}
