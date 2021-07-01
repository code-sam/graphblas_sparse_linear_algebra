use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    BinaryOperator, MatrixColumnExtractor, MatrixTranspose, OperatorOptions, VectorMask,
};
use crate::sparse_matrix::{Size, SparseMatrix};
use crate::sparse_vector::SparseVector;
use crate::util::{ElementIndex, ElementIndexSelector};
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

impl<Matrix, Column> MatrixRowExtractor<Matrix, Column>
where
    Matrix: ValueType,
    Column: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<Column, Column, Column>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let transpose_operator = MatrixTranspose::new(options, None);
        let column_extractor = MatrixColumnExtractor::new(options, accumulator);

        Self {
            transpose_operator,
            column_extractor,
        }
    }

    pub fn apply(
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

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        row_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        row_vector: &mut SparseVector<Column>,
        mask: &VectorMask<MaskValueType, AsBool>,
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

        // let row_index_to_extract = ElementIndex::from(matrix_to_extract_from.row_height()? - row_index_to_extract);

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

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::sparse_matrix::{FromMatrixElementList, MatrixElementList};
    use crate::sparse_vector::GetVectorElementValue;

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
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut column_vector = SparseVector::<u8>::new(&context, &2).unwrap();

        let indices_to_extract: Vec<ElementIndex> = vec![0, 1];
        let indices_to_extract = ElementIndexSelector::Index(&indices_to_extract);

        let extractor = MatrixRowExtractor::new(&OperatorOptions::new_default(), None);

        extractor
            .apply(&matrix, &2, &indices_to_extract, &mut column_vector)
            .unwrap();

        assert_eq!(column_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(column_vector.get_element_value(&0).unwrap(), 3);
        assert_eq!(column_vector.get_element_value(&1).unwrap(), 6);
    }
}
