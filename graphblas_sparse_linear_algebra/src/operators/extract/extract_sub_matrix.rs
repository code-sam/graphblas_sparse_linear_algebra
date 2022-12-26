use std::marker::PhantomData;
use std::ptr;

use crate::collections::sparse_matrix::{
    GraphblasSparseMatrixTrait, SparseMatrix, SparseMatrixTrait,
};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::index::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_trait_for_2_type_data_type_and_all_value_types;
use crate::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_extract,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
implement_trait_for_2_type_data_type_and_all_value_types!(Send, SubMatrixExtractor);
implement_trait_for_2_type_data_type_and_all_value_types!(Sync, SubMatrixExtractor);

#[derive(Debug, Clone)]
pub struct SubMatrixExtractor<Matrix, SubMatrix>
where
    Matrix: ValueType,
    SubMatrix: ValueType,
{
    _matrix: PhantomData<Matrix>,
    _sub_matrix: PhantomData<SubMatrix>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<Matrix, SubMatrix> SubMatrixExtractor<Matrix, SubMatrix>
where
    Matrix: ValueType,
    SubMatrix: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<SubMatrix, SubMatrix, SubMatrix, SubMatrix>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _matrix: PhantomData,
            _sub_matrix: PhantomData,
        }
    }

    pub fn apply(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        rows_to_extract: &ElementIndexSelector, // length must equal row_height of sub_matrix
        columns_to_extract: &ElementIndexSelector, // length must equal column_width of sub_matrix
        sub_matrix: &mut SparseMatrix<SubMatrix>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_extract_from.context();

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
                            sub_matrix.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_extract,
                            column.as_ptr(),
                            number_of_columns_to_extract,
                            self.options,
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            sub_matrix.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            row,
                            number_of_rows_to_extract,
                            column.as_ptr(),
                            number_of_columns_to_extract,
                            self.options,
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            sub_matrix.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_extract,
                            column,
                            number_of_columns_to_extract,
                            self.options,
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            sub_matrix.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            row,
                            number_of_rows_to_extract,
                            column,
                            number_of_columns_to_extract,
                            self.options,
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ref() },
                )?;
            }
        }

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        rows_to_extract: &ElementIndexSelector, // length must equal row_height of sub_matrix
        columns_to_extract: &ElementIndexSelector, // length must equal column_width of sub_matrix
        sub_matrix: &mut SparseMatrix<SubMatrix>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_extract_from.context();

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
                            sub_matrix.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_extract,
                            column.as_ptr(),
                            number_of_columns_to_extract,
                            self.options,
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::Index(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            sub_matrix.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            row,
                            number_of_rows_to_extract,
                            column.as_ptr(),
                            number_of_columns_to_extract,
                            self.options,
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::Index(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            sub_matrix.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            row.as_ptr(),
                            number_of_rows_to_extract,
                            column,
                            number_of_columns_to_extract,
                            self.options,
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ref() },
                )?;
            }
            (
                ElementIndexSelectorGraphblasType::All(row),
                ElementIndexSelectorGraphblasType::All(column),
            ) => {
                context.call(
                    || unsafe {
                        GrB_Matrix_extract(
                            sub_matrix.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            row,
                            number_of_rows_to_extract,
                            column,
                            number_of_columns_to_extract,
                            self.options,
                        )
                    },
                    unsafe { sub_matrix.graphblas_matrix_ref() },
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::collection::Collection;
    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList,
    };
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;

    #[test]
    fn test_matrix_extraction() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (2, 5, 11).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &(10, 15).into(),
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let mut sub_matrix = SparseMatrix::<u8>::new(&context, &(3, 6).into()).unwrap();

        let rows_to_extract: Vec<ElementIndex> = (0..3).collect();
        let rows_to_extract = ElementIndexSelector::Index(&rows_to_extract);
        let columns_to_extract: Vec<ElementIndex> = (0..6).collect();
        let columns_to_extract = ElementIndexSelector::Index(&columns_to_extract);

        let extractor = SubMatrixExtractor::new(&OperatorOptions::new_default(), None);

        extractor
            .apply(
                &matrix,
                &rows_to_extract,
                &columns_to_extract,
                &mut sub_matrix,
            )
            .unwrap();

        assert_eq!(
            sub_matrix.number_of_stored_elements().unwrap(),
            element_list.length()
        );
        assert_eq!(sub_matrix.get_element_value(&(2, 5).into()).unwrap(), 11);

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
                &mut sub_matrix,
            )
            .unwrap();

        assert_eq!(sub_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(sub_matrix.get_element_value(&(1, 5).into()).unwrap(), 0);
        assert_eq!(sub_matrix.get_element_value(&(1, 1).into()).unwrap(), 1);
    }
}
