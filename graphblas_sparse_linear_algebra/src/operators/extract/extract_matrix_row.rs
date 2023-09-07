use crate::collections::sparse_matrix::{GraphblasSparseMatrixTrait, SparseMatrixTrait};
use crate::collections::sparse_vector::SparseVector;
use crate::context::ContextTrait;
use crate::error::SparseLinearAlgebraError;
use crate::index::{ElementIndex, ElementIndexSelector};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::OperatorOptionsTrait;
use crate::operators::{
    extract::{ExtractMatrixColumn, MatrixColumnExtractor},
    options::OperatorOptions,
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
        matrix_to_extract_from: &(impl GraphblasSparseMatrixTrait + ContextTrait + SparseMatrixTrait),
        row_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        accumulator: &impl AccumulatorBinaryOperator<Row>,
        row_vector: &mut SparseVector<Row>,
        mask: &(impl VectorMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<Row: ValueType> ExtractMatrixRow<Row> for MatrixRowExtractor {
    fn apply(
        &self,
        matrix_to_extract_from: &(impl GraphblasSparseMatrixTrait + ContextTrait + SparseMatrixTrait),
        row_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        accumulator: &impl AccumulatorBinaryOperator<Row>,
        row_vector: &mut SparseVector<Row>,
        mask: &(impl VectorMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        // TODO: reduce cost by reusing instance
        let column_extractor = MatrixColumnExtractor::new();

        column_extractor.apply(
            matrix_to_extract_from,
            &row_index_to_extract,
            indices_to_extract,
            accumulator,
            row_vector,
            mask,
            &options.with_negated_transpose_input0(),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, MatrixElementList, SparseMatrix,
    };
    use crate::collections::sparse_vector::operations::GetVectorElementValue;
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::SelectEntireVector;

    #[test]
    fn test_row_extraction() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (2, 0, 3).into(),
            (0, 1, 4).into(),
            (1, 1, 5).into(),
            (2, 1, 6).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &(3, 2).into(),
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut column_vector = SparseVector::<u8>::new(&context, &2).unwrap();

        let indices_to_extract: Vec<ElementIndex> = vec![0, 1];
        let indices_to_extract = ElementIndexSelector::Index(&indices_to_extract);

        let extractor = MatrixRowExtractor::new();

        extractor
            .apply(
                &matrix,
                &2,
                &indices_to_extract,
                &Assignment::<u8>::new(),
                &mut column_vector,
                &SelectEntireVector::new(&context),
                &mut OperatorOptions::new_default(),
            )
            .unwrap();

        assert_eq!(column_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(column_vector.get_element_value_or_default(&0).unwrap(), 3);
        assert_eq!(column_vector.get_element_value_or_default(&1).unwrap(), 6);
    }
}
