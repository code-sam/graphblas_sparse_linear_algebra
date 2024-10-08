use crate::collections::sparse_matrix::operations::GetSparseMatrixSize;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::GetContext;
use crate::error::SparseLinearAlgebraError;
use crate::index::{ElementIndex, ElementIndexSelector};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::extract::{ExtractMatrixColumn, MatrixColumnExtractor};
use crate::operators::mask::VectorMask;
use crate::operators::options::{
    GetOptionsForOperatorWithMatrixArgument, WithTransposeMatrixArgument,
};
use crate::value_type::ValueType;

#[derive(Debug, Clone)]
pub struct MatrixRowExtractor {}

unsafe impl Send for MatrixRowExtractor {}
unsafe impl Sync for MatrixRowExtractor {}

impl MatrixRowExtractor {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ExtractMatrixRow<Row: ValueType> {
    fn apply(
        &self,
        matrix_to_extract_from: &(impl GetGraphblasSparseMatrix + GetSparseMatrixSize + GetContext),
        row_index_to_extract: ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        accumulator: &impl AccumulatorBinaryOperator<Row>,
        row_vector: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &(impl GetOptionsForOperatorWithMatrixArgument + WithTransposeMatrixArgument),
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<Row: ValueType> ExtractMatrixRow<Row> for MatrixRowExtractor {
    fn apply(
        &self,
        matrix_to_extract_from: &(impl GetGraphblasSparseMatrix + GetSparseMatrixSize + GetContext),
        row_index_to_extract: ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        accumulator: &impl AccumulatorBinaryOperator<Row>,
        row_vector: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &(impl GetOptionsForOperatorWithMatrixArgument + WithTransposeMatrixArgument),
    ) -> Result<(), SparseLinearAlgebraError> {
        // TODO: reduce cost by reusing instance
        let column_extractor = MatrixColumnExtractor::new();

        column_extractor.apply(
            matrix_to_extract_from,
            row_index_to_extract,
            indices_to_extract,
            accumulator,
            row_vector,
            mask,
            &options.with_negated_transpose_matrix_argument(),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::FromMatrixElementList;
    use crate::collections::sparse_matrix::{MatrixElementList, SparseMatrix};
    use crate::collections::sparse_vector::operations::GetSparseVectorElementValue;
    use crate::collections::sparse_vector::SparseVector;
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::options::OptionsForOperatorWithMatrixArgument;

    #[test]
    fn test_row_extraction() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (2, 0, 3).into(),
            (0, 1, 4).into(),
            (1, 1, 5).into(),
            (2, 1, 6).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            (3, 2).into(),
            element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut column_vector = SparseVector::<u8>::new(context.clone(), 2).unwrap();

        let indices_to_extract: Vec<ElementIndex> = vec![0, 1];
        let indices_to_extract = ElementIndexSelector::Index(&indices_to_extract);

        let extractor = MatrixRowExtractor::new();

        extractor
            .apply(
                &matrix,
                2,
                &indices_to_extract,
                &Assignment::<u8>::new(),
                &mut column_vector,
                &SelectEntireVector::new(context.clone()),
                &mut OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        assert_eq!(column_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(column_vector.element_value_or_default(0).unwrap(), 3);
        assert_eq!(column_vector.element_value_or_default(1).unwrap(), 6);
    }
}
