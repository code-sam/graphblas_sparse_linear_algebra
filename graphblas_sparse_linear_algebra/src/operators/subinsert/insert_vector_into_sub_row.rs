use std::marker::PhantomData;
use std::ptr;

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GxB_Row_subassign,
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
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_2_type_macro_for_all_value_types_and_untyped_graphblas_function,
    implement_trait_for_2_type_data_type_and_all_value_types,
};
use crate::value_type::{AsBoolean, ConvertScalar, ValueType};

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<MatrixToInsertInto: ValueType, VectorToInsert: ValueType> Send
    for InsertVectorIntoSubRow<MatrixToInsertInto, VectorToInsert>
{
}
unsafe impl<MatrixToInsertInto: ValueType, VectorToInsert: ValueType> Sync
    for InsertVectorIntoSubRow<MatrixToInsertInto, VectorToInsert>
{
}

#[derive(Debug, Clone)]
pub struct InsertVectorIntoSubRow<MatrixToInsertInto: ValueType, VectorToInsert: ValueType> {
    _matrix_to_insert_into: PhantomData<MatrixToInsertInto>,
    _vector_to_insert: PhantomData<VectorToInsert>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<MatrixToInsertInto, VectorToInsert> InsertVectorIntoSubRow<MatrixToInsertInto, VectorToInsert>
where
    MatrixToInsertInto: ValueType,
    VectorToInsert: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<
            &dyn BinaryOperator<
                VectorToInsert,
                MatrixToInsertInto,
                MatrixToInsertInto,
                MatrixToInsertInto,
            >,
        >, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _matrix_to_insert_into: PhantomData,
            _vector_to_insert: PhantomData,
        }
    }
}

pub trait InsertVectorIntoSubRowTrait<MatrixToInsertInto, VectorToInsert>
where
    MatrixToInsertInto: ValueType,
    VectorToInsert: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        row_indices_to_insert_into: &ElementIndexSelector,
        row_to_insert_into: &ElementIndex,
        vector_to_insert: &SparseVector<VectorToInsert>,
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        row_indices_to_insert_into: &ElementIndexSelector,
        row_to_insert_into: &ElementIndex,
        vector_to_insert: &SparseVector<VectorToInsert>,
        mask_for_row_to_insert_into: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<MatrixToInsertInto: ValueType, VectorToInsert: ValueType>
    InsertVectorIntoSubRowTrait<MatrixToInsertInto, VectorToInsert>
    for InsertVectorIntoSubRow<MatrixToInsertInto, VectorToInsert>
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        row_indices_to_insert_into: &ElementIndexSelector,
        row_to_insert_into: &ElementIndex,
        vector_to_insert: &SparseVector<VectorToInsert>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

        let number_of_indices_to_insert_into = row_indices_to_insert_into
            .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
            .to_graphblas_index()?;

        let indices_to_insert_into = row_indices_to_insert_into.to_graphblas_type()?;
        let row_to_insert_into = row_to_insert_into.to_graphblas_index()?;

        match indices_to_insert_into {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GxB_Row_subassign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            vector_to_insert.graphblas_vector(),
                            row_to_insert_into,
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            self.options,
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }

            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GxB_Row_subassign(
                            matrix_to_insert_into.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            vector_to_insert.graphblas_vector(),
                            row_to_insert_into,
                            index,
                            number_of_indices_to_insert_into,
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
    fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        row_indices_to_insert_into: &ElementIndexSelector,
        row_to_insert_into: &ElementIndex,
        vector_to_insert: &SparseVector<VectorToInsert>,
        mask_for_row_to_insert_into: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = matrix_to_insert_into.context();

        let number_of_indices_to_insert_into = row_indices_to_insert_into
            .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
            .to_graphblas_index()?;

        let indices_to_insert_into = row_indices_to_insert_into.to_graphblas_type()?;
        let row_to_insert_into = row_to_insert_into.to_graphblas_index()?;

        match indices_to_insert_into {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(
                    || unsafe {
                        GxB_Row_subassign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_row_to_insert_into.graphblas_vector(),
                            self.accumulator,
                            vector_to_insert.graphblas_vector(),
                            row_to_insert_into,
                            index.as_ptr(),
                            number_of_indices_to_insert_into,
                            self.options,
                        )
                    },
                    unsafe { matrix_to_insert_into.graphblas_matrix_ref() },
                )?;
            }

            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(
                    || unsafe {
                        GxB_Row_subassign(
                            matrix_to_insert_into.graphblas_matrix(),
                            mask_for_row_to_insert_into.graphblas_vector(),
                            self.accumulator,
                            vector_to_insert.graphblas_vector(),
                            row_to_insert_into,
                            index,
                            number_of_indices_to_insert_into,
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

    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };
    use crate::collections::sparse_vector::{FromVectorElementList, VectorElementList};
    use crate::index::ElementIndex;

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
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let element_list_to_insert = VectorElementList::<u8>::from_element_vector(vec![
            (1, 2).into(),
            (2, 3).into(),
            (3, 11).into(),
            // (5, 11).into(),
        ]);

        let vector_to_insert_length: usize = 4;
        let vector_to_insert = SparseVector::<u8>::from_element_list(
            &context,
            &vector_to_insert_length,
            &element_list_to_insert,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let mask_element_list = VectorElementList::<bool>::from_element_vector(vec![
            (1, false).into(),
            (2, true).into(),
            (3, true).into(),
            // (5, true).into(),
        ]);
        let mask = SparseVector::<bool>::from_element_list(
            &context,
            &vector_to_insert_length,
            &mask_element_list,
            &First::<bool, bool, bool, bool>::new(),
        )
        .unwrap();

        let indices_to_insert: Vec<ElementIndex> = (0..vector_to_insert_length).collect();
        let indices_to_insert = ElementIndexSelector::Index(&indices_to_insert);

        let insert_operator = InsertVectorIntoSubRow::new(&OperatorOptions::new_default(), None);

        let row_to_insert_into: ElementIndex = 2;

        insert_operator
            .apply(
                &mut matrix,
                &indices_to_insert,
                &row_to_insert_into,
                &vector_to_insert,
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 6);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
        assert_eq!(matrix.get_element_value(&(1, 2).into()).unwrap(), 1);
        assert_eq!(matrix.get_element_value(&(5, 2).into()).unwrap(), 12);
        assert_eq!(matrix.get_element_value(&(2, 0).into()).unwrap(), 0);
        assert_eq!(matrix.get_element_value(&(2, 4).into()).unwrap(), 0);

        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply_with_mask(
                &mut matrix,
                &indices_to_insert,
                &row_to_insert_into,
                &vector_to_insert,
                &mask,
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 5);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
        assert_eq!(matrix.get_element_value(&(2, 2).into()).unwrap(), 3);
        assert_eq!(matrix.get_element_value(&(1, 2).into()).unwrap(), 1);
        assert_eq!(matrix.get_element_value(&(5, 2).into()).unwrap(), 12);
        assert_eq!(matrix.get_element_value(&(2, 4).into()).unwrap(), 0);
        assert_eq!(matrix.get_element_value(&(2, 1).into()).unwrap(), 0);
    }
}
