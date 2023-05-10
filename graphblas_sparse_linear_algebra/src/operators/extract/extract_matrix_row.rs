use crate::collections::sparse_matrix::{Size, SparseMatrix, SparseMatrixTrait};
use crate::collections::sparse_vector::SparseVector;
use crate::context::ContextTrait;
use crate::error::SparseLinearAlgebraError;
use crate::index::{ElementIndex, ElementIndexSelector};
use crate::operators::binary_operator::{AccumulatorBinaryOperator, Assignment};
use crate::operators::{
    binary_operator::BinaryOperator,
    extract::{ExtractMatrixColumn, MatrixColumnExtractor},
    options::OperatorOptions,
    transpose::{MatrixTranspose, TransposeMatrix},
};
use crate::value_type::{AsBoolean, ValueType};

#[derive(Debug, Clone)]
pub struct MatrixRowExtractor<Matrix, Column>
where
    Matrix: ValueType,
    Column: ValueType,
{
    transpose_operator: MatrixTranspose<Matrix, Matrix>,
    column_extractor: MatrixColumnExtractor<Matrix, Column>,
}

unsafe impl<Matrix: ValueType, Column: ValueType> Send for MatrixRowExtractor<Matrix, Column> {}
unsafe impl<Matrix: ValueType, Column: ValueType> Sync for MatrixRowExtractor<Matrix, Column> {}

impl<Matrix, Column> MatrixRowExtractor<Matrix, Column>
where
    Matrix: ValueType,
    Column: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<Column>,
    ) -> Self {
        let transpose_operator = MatrixTranspose::new(
            options,
            &Assignment::<Matrix>::new(),
        );
        let column_extractor = MatrixColumnExtractor::new(options, accumulator);

        Self {
            transpose_operator,
            column_extractor,
        }
    }
}

pub trait ExtractMatrixRow<Matrix: ValueType, Column: ValueType> {
    fn apply(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        row_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        row_vector: &mut SparseVector<Column>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        row_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        row_vector: &mut SparseVector<Column>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<Matrix: ValueType, Column: ValueType> ExtractMatrixRow<Matrix, Column>
    for MatrixRowExtractor<Matrix, Column>
{
    fn apply(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        row_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        row_vector: &mut SparseVector<Column>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let _context = matrix_to_extract_from.context();

        let size_of_transposed_matrix: Size = (
            matrix_to_extract_from.column_width()?,
            matrix_to_extract_from.row_height()?,
        )
            .into();
        // creating a new matrix, instead of cloning, requires to specify if if the value type is built-in or custom, as this is required for the SparseMatrix::new() constructor.
        // let mut transposed_matrix = SparseMatrix::<ValueType>::new(&context, &size_of_transposed_matrix)?;
        let mut transposed_matrix = matrix_to_extract_from.clone();
        transposed_matrix.resize(&size_of_transposed_matrix)?;

        self.transpose_operator
            .apply(&matrix_to_extract_from, &mut transposed_matrix)?;

        self.column_extractor.apply(
            &transposed_matrix,
            row_index_to_extract,
            indices_to_extract,
            row_vector,
        )?;

        Ok(())
    }

    fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        row_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        row_vector: &mut SparseVector<Column>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let _context = matrix_to_extract_from.context();

        let size_of_transposed_matrix: Size = (
            matrix_to_extract_from.column_width()?,
            matrix_to_extract_from.row_height()?,
        )
            .into();
        let mut transposed_matrix = SparseMatrix::<Matrix>::new(
            matrix_to_extract_from.context_ref(),
            &size_of_transposed_matrix,
        )?;
        transposed_matrix.resize(&size_of_transposed_matrix)?;

        self.transpose_operator
            .apply(&matrix_to_extract_from, &mut transposed_matrix)?;

        self.column_extractor.apply_with_mask(
            &matrix_to_extract_from,
            &row_index_to_extract,
            indices_to_extract,
            row_vector,
            mask,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::{FromMatrixElementList, MatrixElementList};
    use crate::collections::sparse_vector::GetVectorElementValue;
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;

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
            &context.clone(),
            &(3, 2).into(),
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut column_vector = SparseVector::<u8>::new(&context, &2).unwrap();

        let indices_to_extract: Vec<ElementIndex> = vec![0, 1];
        let indices_to_extract = ElementIndexSelector::Index(&indices_to_extract);

        let extractor = MatrixRowExtractor::new(
            &OperatorOptions::new_default(),
            &Assignment::<u8>::new(),
        );

        extractor
            .apply(&matrix, &2, &indices_to_extract, &mut column_vector)
            .unwrap();

        assert_eq!(column_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(column_vector.get_element_value_or_default(&0).unwrap(), 3);
        assert_eq!(column_vector.get_element_value_or_default(&1).unwrap(), 6);
    }
}
