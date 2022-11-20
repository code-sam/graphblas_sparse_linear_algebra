use std::marker::PhantomData;
use std::ptr;

use crate::collections::collection::Collection;
use crate::collections::sparse_matrix::SparseMatrix;
use crate::collections::sparse_vector::SparseVector;
use crate::context::CallGraphBlasContext;
use crate::error::SparseLinearAlgebraError;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::util::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::value_types::utilities_to_implement_traits_for_all_value_types::implement_trait_for_2_type_data_type_and_all_value_types;
use crate::value_types::value_type::{AsBoolean, BuiltInValueType, ValueType};

use crate::bindings_to_graphblas_implementation::{GrB_BinaryOp, GrB_Col_extract, GrB_Descriptor};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
implement_trait_for_2_type_data_type_and_all_value_types!(Send, MatrixColumnExtractor);
implement_trait_for_2_type_data_type_and_all_value_types!(Sync, MatrixColumnExtractor);

#[derive(Debug, Clone)]
pub struct MatrixColumnExtractor<Matrix, Column>
where
    Matrix: ValueType + BuiltInValueType,
    Column: ValueType + BuiltInValueType,
{
    _matrix: PhantomData<Matrix>,
    _column: PhantomData<Column>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<Matrix, Column> MatrixColumnExtractor<Matrix, Column>
where
    Matrix: ValueType + BuiltInValueType,
    Column: ValueType + BuiltInValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<Column, Column, Column>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
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
            _column: PhantomData,
        }
    }

    pub fn apply(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        column_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        column_vector: &mut SparseVector<Column>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_extract_from.context();

        let number_of_indices_to_extract: ElementIndex;
        match indices_to_extract {
            ElementIndexSelector::Index(indices) => number_of_indices_to_extract = indices.len(),
            ElementIndexSelector::All => {
                number_of_indices_to_extract = matrix_to_extract_from.row_height()?
            }
        }
        let number_of_indices_to_extract = number_of_indices_to_extract.to_graphblas_index()?;

        let indices_to_extract = indices_to_extract.to_graphblas_type()?;

        let column_index_to_extract = column_index_to_extract.to_graphblas_index()?;

        match indices_to_extract {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GrB_Col_extract(
                            column_vector.graphblas_vector(),
                            ptr::null_mut(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            index.as_ptr(),
                            number_of_indices_to_extract,
                            column_index_to_extract,
                            self.options,
                        )
                    },
                    column_vector.graphblas_vector_ref(),
                )?;
            }
            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GrB_Col_extract(
                            column_vector.graphblas_vector(),
                            ptr::null_mut(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            index,
                            number_of_indices_to_extract,
                            column_index_to_extract,
                            self.options,
                        )
                    },
                    column_vector.graphblas_vector_ref(),
                )?;
            }
        }

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        matrix_to_extract_from: &SparseMatrix<Matrix>,
        column_index_to_extract: &ElementIndex,
        indices_to_extract: &ElementIndexSelector,
        column_vector: &mut SparseVector<Column>,
        mask: &SparseVector<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_extract_from.context();

        let number_of_indices_to_extract: ElementIndex;
        match indices_to_extract {
            ElementIndexSelector::Index(indices) => number_of_indices_to_extract = indices.len(),
            ElementIndexSelector::All => {
                number_of_indices_to_extract = matrix_to_extract_from.row_height()?
            }
        }
        let number_of_indices_to_extract = number_of_indices_to_extract.to_graphblas_index()?;

        let indices_to_extract = indices_to_extract.to_graphblas_type()?;

        let column_index_to_extract = column_index_to_extract.to_graphblas_index()?;

        match indices_to_extract {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GrB_Col_extract(
                            column_vector.graphblas_vector(),
                            mask.graphblas_vector(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            index.as_ptr(),
                            number_of_indices_to_extract,
                            column_index_to_extract,
                            self.options,
                        )
                    },
                    column_vector.graphblas_vector_ref(),
                )?;
            }
            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GrB_Col_extract(
                            column_vector.graphblas_vector(),
                            mask.graphblas_vector(),
                            self.accumulator,
                            matrix_to_extract_from.graphblas_matrix(),
                            index,
                            number_of_indices_to_extract,
                            column_index_to_extract,
                            self.options,
                        )
                    },
                    column_vector.graphblas_vector_ref(),
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::{FromMatrixElementList, MatrixElementList};
    use crate::collections::sparse_vector::GetVectorElementValue;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;

    #[test]
    fn test_column_extraction() {
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

        let indices_to_extract: Vec<ElementIndex> = vec![0, 2];
        let indices_to_extract = ElementIndexSelector::Index(&indices_to_extract);

        let extractor = MatrixColumnExtractor::new(&OperatorOptions::new_default(), None);

        extractor
            .apply(&matrix, &0, &indices_to_extract, &mut column_vector)
            .unwrap();

        assert_eq!(column_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(column_vector.get_element_value(&0).unwrap(), 1);
        // assert_eq!(column_vector.get_element_value(&1).unwrap(), 0);
        assert_eq!(column_vector.get_element_value(&1).unwrap(), 3);
    }
}
