use std::ptr;

use std::marker::PhantomData;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator, mask::MatrixMask, options::OperatorOptions,
};
use crate::value_types::sparse_matrix::SparseMatrix;

use crate::util::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::value_types::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GxB_Matrix_subassign,
};

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertMatrixIntoSubMatrix<bool,bool> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<u8,u8> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<u16,u16> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<u32,u32> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<u64,u64> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<i8,i8> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<i16,i16> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<i32,i32> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<i64,i64> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<f32,f32> {}
unsafe impl Send for InsertMatrixIntoSubMatrix<f64,f64> {}

unsafe impl Sync for InsertMatrixIntoSubMatrix<bool,bool> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<u8,u8> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<u16,u16> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<u32,u32> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<u64,u64> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<i8,i8> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<i16,i16> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<i32,i32> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<i64,i64> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<f32,f32> {}
unsafe impl Sync for InsertMatrixIntoSubMatrix<f64,f64> {}

#[derive(Debug, Clone)]
pub struct InsertMatrixIntoSubMatrix<MatrixToInsertInto: ValueType, MatrixToInsert: ValueType> {
    _matrix_to_insert_into: PhantomData<MatrixToInsertInto>,
    _matrix_to_insert: PhantomData<MatrixToInsert>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<MatrixToInsertInto, MatrixToInsert>
    InsertMatrixIntoSubMatrix<MatrixToInsertInto, MatrixToInsert>
where
    MatrixToInsertInto: ValueType,
    MatrixToInsert: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<
            &dyn BinaryOperator<MatrixToInsert, MatrixToInsertInto, MatrixToInsertInto>,
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
            _matrix_to_insert: PhantomData,
        }
    }
}

pub trait InsertMatrixIntoSubMatrixTrait<MatrixToInsertInto, MatrixToInsert>
where
    MatrixToInsertInto: ValueType,
    MatrixToInsert: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &SparseMatrix<MatrixToInsert>,
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
        columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
        matrix_to_insert: &SparseMatrix<MatrixToInsert>,
        mask_for_matrix_to_insert_into: &MatrixMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_insert_matrix_into_sub_matrix_trait {
    (
        $value_type_matrix_to_insert_into:ty, $value_type_matrix_to_insert:ty, $graphblas_insert_function:ident
    ) => {
        impl
            InsertMatrixIntoSubMatrixTrait<
                $value_type_matrix_to_insert_into,
                $value_type_matrix_to_insert,
            >
            for InsertMatrixIntoSubMatrix<
                $value_type_matrix_to_insert_into,
                $value_type_matrix_to_insert,
            >
        {
            /// replace option applies to entire matrix_to_insert_to
            fn apply(
                &self,
                matrix_to_insert_into: &mut SparseMatrix<$value_type_matrix_to_insert_into>,
                rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
                columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
                matrix_to_insert: &SparseMatrix<$value_type_matrix_to_insert>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = matrix_to_insert_into.context();

                let number_of_rows_to_insert_into = rows_to_insert_into
                    .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
                    .to_graphblas_index()?;

                let number_of_columns_to_insert_into = columns_to_insert_into
                    .number_of_selected_elements(matrix_to_insert_into.column_width()?)?
                    .to_graphblas_index()?;

                let rows_to_insert_into = rows_to_insert_into.to_graphblas_type()?;
                let columns_to_insert_into = columns_to_insert_into.to_graphblas_type()?;

                let matrix_to_insert_into_with_write_lock =
                    matrix_to_insert_into.get_write_lock()?;
                let matrix_to_insert_with_read_lock = matrix_to_insert.get_read_lock()?;

                match (rows_to_insert_into, columns_to_insert_into) {
                    (
                        ElementIndexSelectorGraphblasType::Index(row),
                        ElementIndexSelectorGraphblasType::Index(column),
                    ) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                *matrix_to_insert_into_with_write_lock,
                                ptr::null_mut(),
                                self.accumulator,
                                *matrix_to_insert_with_read_lock,
                                row.as_ptr(),
                                number_of_rows_to_insert_into,
                                column.as_ptr(),
                                number_of_columns_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                    (
                        ElementIndexSelectorGraphblasType::All(row),
                        ElementIndexSelectorGraphblasType::Index(column),
                    ) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                *matrix_to_insert_into_with_write_lock,
                                ptr::null_mut(),
                                self.accumulator,
                                *matrix_to_insert_with_read_lock,
                                row,
                                number_of_rows_to_insert_into,
                                column.as_ptr(),
                                number_of_columns_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                    (
                        ElementIndexSelectorGraphblasType::Index(row),
                        ElementIndexSelectorGraphblasType::All(column),
                    ) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                *matrix_to_insert_into_with_write_lock,
                                ptr::null_mut(),
                                self.accumulator,
                                *matrix_to_insert_with_read_lock,
                                row.as_ptr(),
                                number_of_rows_to_insert_into,
                                column,
                                number_of_columns_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                    (
                        ElementIndexSelectorGraphblasType::All(row),
                        ElementIndexSelectorGraphblasType::All(column),
                    ) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                *matrix_to_insert_into_with_write_lock,
                                ptr::null_mut(),
                                self.accumulator,
                                *matrix_to_insert_with_read_lock,
                                row,
                                number_of_rows_to_insert_into,
                                column,
                                number_of_columns_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                }

                Ok(())
            }

            /// mask and replace option apply to entire matrix_to_insert_to
            fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
                &self,
                matrix_to_insert_into: &mut SparseMatrix<$value_type_matrix_to_insert_into>,
                rows_to_insert_into: &ElementIndexSelector, // length must equal row_height of matrix_to_insert
                columns_to_insert_into: &ElementIndexSelector, // length must equal column_width of matrix_to_insert
                matrix_to_insert: &SparseMatrix<$value_type_matrix_to_insert>,
                mask_for_matrix_to_insert_into: &MatrixMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = matrix_to_insert_into.context();

                let number_of_rows_to_insert_into = rows_to_insert_into
                    .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
                    .to_graphblas_index()?;

                let number_of_columns_to_insert_into = columns_to_insert_into
                    .number_of_selected_elements(matrix_to_insert_into.column_width()?)?
                    .to_graphblas_index()?;

                let rows_to_insert_into = rows_to_insert_into.to_graphblas_type()?;
                let columns_to_insert_into = columns_to_insert_into.to_graphblas_type()?;

                let matrix_to_insert_into_with_write_lock =
                    matrix_to_insert_into.get_write_lock()?;
                let mask_for_matrix_to_insert_into_with_read_lock =
                    mask_for_matrix_to_insert_into.get_read_lock()?;
                let matrix_to_insert_with_read_lock = matrix_to_insert.get_read_lock()?;

                match (rows_to_insert_into, columns_to_insert_into) {
                    (
                        ElementIndexSelectorGraphblasType::Index(row),
                        ElementIndexSelectorGraphblasType::Index(column),
                    ) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                *matrix_to_insert_into_with_write_lock,
                                *mask_for_matrix_to_insert_into_with_read_lock,
                                self.accumulator,
                                *matrix_to_insert_with_read_lock,
                                row.as_ptr(),
                                number_of_rows_to_insert_into,
                                column.as_ptr(),
                                number_of_columns_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                    (
                        ElementIndexSelectorGraphblasType::All(row),
                        ElementIndexSelectorGraphblasType::Index(column),
                    ) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                *matrix_to_insert_into_with_write_lock,
                                *mask_for_matrix_to_insert_into_with_read_lock,
                                self.accumulator,
                                *matrix_to_insert_with_read_lock,
                                row,
                                number_of_rows_to_insert_into,
                                column.as_ptr(),
                                number_of_columns_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                    (
                        ElementIndexSelectorGraphblasType::Index(row),
                        ElementIndexSelectorGraphblasType::All(column),
                    ) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                *matrix_to_insert_into_with_write_lock,
                                *mask_for_matrix_to_insert_into_with_read_lock,
                                self.accumulator,
                                *matrix_to_insert_with_read_lock,
                                row.as_ptr(),
                                number_of_rows_to_insert_into,
                                column,
                                number_of_columns_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                    (
                        ElementIndexSelectorGraphblasType::All(row),
                        ElementIndexSelectorGraphblasType::All(column),
                    ) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                *matrix_to_insert_into_with_write_lock,
                                *mask_for_matrix_to_insert_into_with_read_lock,
                                self.accumulator,
                                *matrix_to_insert_with_read_lock,
                                row,
                                number_of_rows_to_insert_into,
                                column,
                                number_of_columns_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                }

                Ok(())
            }
        }
    };
}

implement_insert_matrix_into_sub_matrix_trait!(bool, bool, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(u8, u8, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(u16, u16, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(u32, u32, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(u64, u64, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(i8, i8, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(i16, i16, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(i32, i32, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(i64, i64, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(f32, f32, GxB_Matrix_subassign);
implement_insert_matrix_into_sub_matrix_trait!(f64, f64, GxB_Matrix_subassign);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;

    use crate::util::ElementIndex;
    use crate::value_types::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };

    #[test]
    fn test_insert_matrix_into_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (9, 5, 11).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let element_list_to_insert = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 2).into(),
            (1, 1, 3).into(),
            (0, 1, 11).into(),
            (1, 0, 12).into(),
        ]);

        let matrix_size_to_insert: Size = (2, 2).into();
        let matrix_to_insert = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size_to_insert,
            &element_list_to_insert,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mask_element_list = MatrixElementList::<bool>::from_element_vector(vec![
            (0, 0, true).into(),
            // (1, 1, true).into(),
            (1, 0, true).into(),
            (0, 1, true).into(),
        ]);
        let mask = SparseMatrix::<bool>::from_element_list(
            &context,
            &matrix_size_to_insert,
            &mask_element_list,
            &First::<bool, bool, bool>::new(),
        )
        .unwrap();

        let rows_to_insert: Vec<ElementIndex> = (0..2).collect();
        let rows_to_insert = ElementIndexSelector::Index(&rows_to_insert);
        let columns_to_insert: Vec<ElementIndex> = (0..2).collect();
        let columns_to_insert = ElementIndexSelector::Index(&columns_to_insert);

        let insert_operator = InsertMatrixIntoSubMatrix::new(&OperatorOptions::new_default(), None);

        insert_operator
            .apply(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &matrix_to_insert,
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 7);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), 2);
        assert_eq!(matrix.get_element_value(&(1, 1).into()).unwrap(), 3);
        assert_eq!(matrix.get_element_value(&(2, 2).into()).unwrap(), 2);
        assert_eq!(matrix.get_element_value(&(2, 4).into()).unwrap(), 10);
        assert_eq!(matrix.get_element_value(&(9, 5).into()).unwrap(), 11);

        let mut matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply_with_mask(
                &mut matrix,
                &rows_to_insert,
                &columns_to_insert,
                &matrix_to_insert,
                &mask.into(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 7);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), 2);
        assert_eq!(matrix.get_element_value(&(2, 2).into()).unwrap(), 2);
        assert_eq!(matrix.get_element_value(&(2, 4).into()).unwrap(), 10);
        assert_eq!(matrix.get_element_value(&(2, 5).into()).unwrap(), 0);
        assert_eq!(matrix.get_element_value(&(1, 1).into()).unwrap(), 1);
    }
}
