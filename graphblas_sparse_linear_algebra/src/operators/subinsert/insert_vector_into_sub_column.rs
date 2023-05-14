use std::marker::PhantomData;
use std::ptr;

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GxB_Col_subassign,
};
use crate::collections::sparse_matrix::{
    GraphblasSparseMatrixTrait, SparseMatrix, SparseMatrixTrait,
};
use crate::collections::sparse_vector::{GraphblasSparseVectorTrait, SparseVector};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::index::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::{AsBoolean, ValueType};

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<MatrixToInsertInto: ValueType> Send for InsertVectorIntoSubColumn<MatrixToInsertInto> {}
unsafe impl<MatrixToInsertInto: ValueType> Sync for InsertVectorIntoSubColumn<MatrixToInsertInto> {}

#[derive(Debug, Clone)]
pub struct InsertVectorIntoSubColumn<MatrixToInsertInto: ValueType> {
    _matrix_to_insert_into: PhantomData<MatrixToInsertInto>,

    accumulator: GrB_BinaryOp,
    options: GrB_Descriptor,
}

impl<MatrixToInsertInto> InsertVectorIntoSubColumn<MatrixToInsertInto>
where
    MatrixToInsertInto: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<MatrixToInsertInto>,
    ) -> Self {
        Self {
            accumulator: accumulator.accumulator_graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _matrix_to_insert_into: PhantomData,
        }
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
        vector_to_insert: &(impl GraphblasSparseVectorTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        column_indices_to_insert_into: &ElementIndexSelector,
        column_to_insert_into: &ElementIndex,
        vector_to_insert: &(impl GraphblasSparseVectorTrait + ContextTrait),
        mask_for_vector_to_insert_into: &(impl GraphblasSparseVectorTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<MatrixToInsertInto: ValueType> InsertVectorIntoSubColumnTrait<MatrixToInsertInto>
    for InsertVectorIntoSubColumn<MatrixToInsertInto>
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        column_indices_to_insert_into: &ElementIndexSelector,
        column_to_insert_into: &ElementIndex,
        vector_to_insert: &(impl GraphblasSparseVectorTrait + ContextTrait),
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
                            self.accumulator,
                            vector_to_insert.graphblas_vector(),
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            column_to_insert_into,
                            self.options,
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
                            self.accumulator,
                            vector_to_insert.graphblas_vector(),
                            index,
                            number_of_indices_to_insert_into,
                            column_to_insert_into,
                            self.options,
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
        vector_to_insert: &(impl GraphblasSparseVectorTrait + ContextTrait),
        mask_for_column_to_insert_into: &(impl GraphblasSparseVectorTrait + ContextTrait),
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
                            self.accumulator,
                            vector_to_insert.graphblas_vector(),
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            column_to_insert_into,
                            self.options,
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
                            self.accumulator,
                            vector_to_insert.graphblas_vector(),
                            index,
                            number_of_indices_to_insert_into,
                            column_to_insert_into,
                            self.options,
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

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };
    use crate::collections::sparse_vector::FromVectorElementList;
    use crate::collections::sparse_vector::VectorElementList;
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::index::ElementIndex;
    use crate::operators::binary_operator::{Assignment, First};

    #[test]
    fn test_insert_vector_into_column() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

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

        let insert_operator =
            InsertVectorIntoSubColumn::new(&OperatorOptions::new_default(), &Assignment::new());

        let column_to_insert_into: ElementIndex = 2;

        insert_operator
            .apply(
                &mut matrix,
                &indices_to_insert,
                &column_to_insert_into,
                &vector_to_insert,
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), None);
        assert_eq!(
            matrix.get_element_value_or_default(&(1, 2).into()).unwrap(),
            2
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(5, 2).into()).unwrap(),
            12
        );

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
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), None);
        assert_eq!(
            matrix.get_element_value_or_default(&(2, 2).into()).unwrap(),
            3
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(1, 2).into()).unwrap(),
            1
        );
        assert_eq!(
            matrix.get_element_value_or_default(&(5, 2).into()).unwrap(),
            12
        );
    }
}
