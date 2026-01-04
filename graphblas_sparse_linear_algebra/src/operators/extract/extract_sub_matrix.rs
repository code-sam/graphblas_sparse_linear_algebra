use crate::collections::sparse_matrix::operations::GetSparseMatrixSize;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::CallGraphBlasContext;
use crate::error::SparseLinearAlgebraError;
use crate::index::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::MatrixMask;
use crate::operators::options::GetOptionsForOperatorWithMatrixArgument;
use crate::value_type::ValueType;

use crate::graphblas_bindings::GrB_Matrix_extract;

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for SubMatrixExtractor {}
unsafe impl Sync for SubMatrixExtractor {}

#[derive(Debug, Clone)]
pub struct SubMatrixExtractor {}

impl SubMatrixExtractor {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ExtractSubMatrix<SubMatrix: ValueType> {
    fn apply(
        &self,
        matrix_to_extract_from: &(impl GetGraphblasSparseMatrix + GetSparseMatrixSize),
        rows_to_extract: &ElementIndexSelector, // length must equal row_height of sub_matrix
        columns_to_extract: &ElementIndexSelector, // length must equal column_width of sub_matrix
        accumulator: &impl AccumulatorBinaryOperator<SubMatrix>,
        sub_matrix: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArgument,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<SubMatrix: ValueType> ExtractSubMatrix<SubMatrix> for SubMatrixExtractor {
    fn apply(
        &self,
        matrix_to_extract_from: &(impl GetGraphblasSparseMatrix + GetSparseMatrixSize),
        rows_to_extract: &ElementIndexSelector, // length must equal row_height of sub_matrix
        columns_to_extract: &ElementIndexSelector, // length must equal column_width of sub_matrix
        accumulator: &impl AccumulatorBinaryOperator<SubMatrix>,
        sub_matrix: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOptionsForOperatorWithMatrixArgument,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_extract_from.context_ref();

        let number_of_rows_to_extract: ElementIndex;
        match rows_to_extract {
            ElementIndexSelector::Index(indices) => number_of_rows_to_extract = indices.len(),
            ElementIndexSelector::All => {
                number_of_rows_to_extract = matrix_to_extract_from.row_height()?
            }
        }
        let number_of_rows_to_extract = number_of_rows_to_extract.to_graphblas_index()?;

        let number_of_columns_to_extract: ElementIndex;
        match columns_to_extract {
            ElementIndexSelector::Index(indices) => number_of_columns_to_extract = indices.len(),
            ElementIndexSelector::All => {
                number_of_columns_to_extract = matrix_to_extract_from.column_width()?
            }
        }
        let number_of_columns_to_extract = number_of_columns_to_extract.to_graphblas_index()?;

        let rows_to_extract = rows_to_extract.to_graphblas_type()?;
        let columns_to_extract = columns_to_extract.to_graphblas_type()?;

        match (rows_to_extract, columns_to_extract) {
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            GetGraphblasSparseMatrix::graphblas_matrix_ptr(sub_matrix),
                            mask.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_extract_from.graphblas_matrix_ptr(),
                            row.as_ptr(),
                            number_of_rows_to_extract,
                            column.as_ptr(),
                            number_of_columns_to_extract,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ptr_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            GetGraphblasSparseMatrix::graphblas_matrix_ptr(sub_matrix),
                            mask.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_extract_from.graphblas_matrix_ptr(),
                            row,
                            number_of_rows_to_extract,
                            column.as_ptr(),
                            number_of_columns_to_extract,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ptr_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            GetGraphblasSparseMatrix::graphblas_matrix_ptr(sub_matrix),
                            mask.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_extract_from.graphblas_matrix_ptr(),
                            row.as_ptr(),
                            number_of_rows_to_extract,
                            column,
                            number_of_columns_to_extract,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ptr_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            GetGraphblasSparseMatrix::graphblas_matrix_ptr(sub_matrix),
                            mask.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            matrix_to_extract_from.graphblas_matrix_ptr(),
                            row,
                            number_of_rows_to_extract,
                            column,
                            number_of_columns_to_extract,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ptr_ref() },
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
    use crate::collections::sparse_matrix::{MatrixElementList, SparseMatrix};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::SelectEntireMatrix;
    use crate::operators::options::OptionsForOperatorWithMatrixArgument;

    #[test]
    fn test_matrix_extraction() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (2, 5, 11).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            (10, 15).into(),
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        let mut sub_matrix = SparseMatrix::<u8>::new(context.clone(), (3, 6).into()).unwrap();

        let rows_to_extract: Vec<ElementIndex> = (0..3).collect();
        let rows_to_extract = ElementIndexSelector::Index(&rows_to_extract);
        let columns_to_extract: Vec<ElementIndex> = (0..6).collect();
        let columns_to_extract = ElementIndexSelector::Index(&columns_to_extract);

        let extractor = SubMatrixExtractor::new();

        extractor
            .apply(
                &matrix,
                &rows_to_extract,
                &columns_to_extract,
                &Assignment::<u8>::new(),
                &mut sub_matrix,
                &SelectEntireMatrix::new(context.clone()),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        assert_eq!(
            sub_matrix.number_of_stored_elements().unwrap(),
            element_list.length()
        );
        assert_eq!(sub_matrix.element_value_or_default(2, 5).unwrap(), 11);

        // Test extraction of suplicate rowsRowOrColumnSelector
        let rows_to_extract = vec![1, 1, 1];
        let rows_to_extract = ElementIndexSelector::Index(&rows_to_extract);
        let columns_to_extract: Vec<ElementIndex> = (0..6).collect();
        let columns_to_extract = ElementIndexSelector::Index(&columns_to_extract);

        extractor
            .apply(
                &matrix,
                &rows_to_extract,
                &columns_to_extract,
                &Assignment::<u8>::new(),
                &mut sub_matrix,
                &SelectEntireMatrix::new(context.clone()),
                &OptionsForOperatorWithMatrixArgument::new_default(),
            )
            .unwrap();

        assert_eq!(sub_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(sub_matrix.element_value(1, 5).unwrap(), None);
        assert_eq!(sub_matrix.element_value_or_default(1, 1).unwrap(), 1);
    }
}
