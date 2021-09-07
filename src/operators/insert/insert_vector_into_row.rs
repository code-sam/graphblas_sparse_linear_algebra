use std::ptr;

use std::marker::PhantomData;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator, mask::VectorMask, options::OperatorOptions,
};
use crate::util::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::value_types::sparse_matrix::SparseMatrix;
use crate::value_types::sparse_vector::SparseVector;
use crate::value_types::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{GrB_BinaryOp, GrB_Descriptor, GrB_Row_assign};

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for InsertVectorIntoRow<bool, bool> {}
unsafe impl Send for InsertVectorIntoRow<u8, u8> {}
unsafe impl Send for InsertVectorIntoRow<u16, u16> {}
unsafe impl Send for InsertVectorIntoRow<u32, u32> {}
unsafe impl Send for InsertVectorIntoRow<u64, u64> {}
unsafe impl Send for InsertVectorIntoRow<i8, i8> {}
unsafe impl Send for InsertVectorIntoRow<i16, i16> {}
unsafe impl Send for InsertVectorIntoRow<i32, i32> {}
unsafe impl Send for InsertVectorIntoRow<i64, i64> {}
unsafe impl Send for InsertVectorIntoRow<f32, f32> {}
unsafe impl Send for InsertVectorIntoRow<f64, f64> {}

unsafe impl Sync for InsertVectorIntoRow<bool, bool> {}
unsafe impl Sync for InsertVectorIntoRow<u8, u8> {}
unsafe impl Sync for InsertVectorIntoRow<u16, u16> {}
unsafe impl Sync for InsertVectorIntoRow<u32, u32> {}
unsafe impl Sync for InsertVectorIntoRow<u64, u64> {}
unsafe impl Sync for InsertVectorIntoRow<i8, i8> {}
unsafe impl Sync for InsertVectorIntoRow<i16, i16> {}
unsafe impl Sync for InsertVectorIntoRow<i32, i32> {}
unsafe impl Sync for InsertVectorIntoRow<i64, i64> {}
unsafe impl Sync for InsertVectorIntoRow<f32, f32> {}
unsafe impl Sync for InsertVectorIntoRow<f64, f64> {}

#[derive(Debug, Clone)]
pub struct InsertVectorIntoRow<MatrixToInsertInto: ValueType, VectorToInsert: ValueType> {
    _matrix_to_insert_into: PhantomData<MatrixToInsertInto>,
    _vector_to_insert: PhantomData<VectorToInsert>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<MatrixToInsertInto, VectorToInsert> InsertVectorIntoRow<MatrixToInsertInto, VectorToInsert>
where
    MatrixToInsertInto: ValueType,
    VectorToInsert: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<
            &dyn BinaryOperator<VectorToInsert, MatrixToInsertInto, MatrixToInsertInto>,
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

pub trait InsertVectorIntoRowTrait<MatrixToInsertInto, VectorToInsert>
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
    fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        matrix_to_insert_into: &mut SparseMatrix<MatrixToInsertInto>,
        row_indices_to_insert_into: &ElementIndexSelector,
        row_to_insert_into: &ElementIndex,
        vector_to_insert: &SparseVector<VectorToInsert>,
        mask_for_row_to_insert_into: &VectorMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_insert_vector_into_row_trait {
    (
        $value_type_matrix_to_insert_into:ty, $value_type_vector_to_insert:ty, $graphblas_insert_function:ident
    ) => {
        impl
            InsertVectorIntoRowTrait<
                $value_type_matrix_to_insert_into,
                $value_type_vector_to_insert,
            >
            for InsertVectorIntoRow<$value_type_matrix_to_insert_into, $value_type_vector_to_insert>
        {
            /// replace option applies to entire matrix_to_insert_to
            fn apply(
                &self,
                matrix_to_insert_into: &mut SparseMatrix<$value_type_matrix_to_insert_into>,
                row_indices_to_insert_into: &ElementIndexSelector,
                row_to_insert_into: &ElementIndex,
                vector_to_insert: &SparseVector<$value_type_vector_to_insert>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = matrix_to_insert_into.context();

                let number_of_indices_to_insert_into = row_indices_to_insert_into
                    .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
                    .to_graphblas_index()?;

                let indices_to_insert_into = row_indices_to_insert_into.to_graphblas_type()?;
                let row_to_insert_into = row_to_insert_into.to_graphblas_index()?;

                match indices_to_insert_into {
                    ElementIndexSelectorGraphblasType::Index(index) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                matrix_to_insert_into.graphblas_matrix(),
                                ptr::null_mut(),
                                self.accumulator,
                                vector_to_insert.graphblas_vector(),
                                row_to_insert_into,
                                index.as_ptr(),
                                number_of_indices_to_insert_into,
                                self.options,
                            )
                        })?;
                    }

                    ElementIndexSelectorGraphblasType::All(index) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                matrix_to_insert_into.graphblas_matrix(),
                                ptr::null_mut(),
                                self.accumulator,
                                vector_to_insert.graphblas_vector(),
                                row_to_insert_into,
                                index,
                                number_of_indices_to_insert_into,
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
                row_indices_to_insert_into: &ElementIndexSelector,
                row_to_insert_into: &ElementIndex,
                vector_to_insert: &SparseVector<$value_type_vector_to_insert>,
                mask_for_row_to_insert_into: &VectorMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = matrix_to_insert_into.context();

                let number_of_indices_to_insert_into = row_indices_to_insert_into
                    .number_of_selected_elements(matrix_to_insert_into.row_height()?)?
                    .to_graphblas_index()?;

                let indices_to_insert_into = row_indices_to_insert_into.to_graphblas_type()?;
                let row_to_insert_into = row_to_insert_into.to_graphblas_index()?;

                match indices_to_insert_into {
                    ElementIndexSelectorGraphblasType::Index(index) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                matrix_to_insert_into.graphblas_matrix(),
                                mask_for_row_to_insert_into.graphblas_vector(),
                                self.accumulator,
                                vector_to_insert.graphblas_vector(),
                                row_to_insert_into,
                                index.as_ptr(),
                                number_of_indices_to_insert_into,
                                self.options,
                            )
                        })?;
                    }

                    ElementIndexSelectorGraphblasType::All(index) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                matrix_to_insert_into.graphblas_matrix(),
                                mask_for_row_to_insert_into.graphblas_vector(),
                                self.accumulator,
                                vector_to_insert.graphblas_vector(),
                                row_to_insert_into,
                                index,
                                number_of_indices_to_insert_into,
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

implement_insert_vector_into_row_trait!(bool, bool, GrB_Row_assign);
implement_insert_vector_into_row_trait!(u8, u8, GrB_Row_assign);
implement_insert_vector_into_row_trait!(u16, u16, GrB_Row_assign);
implement_insert_vector_into_row_trait!(u32, u32, GrB_Row_assign);
implement_insert_vector_into_row_trait!(u64, u64, GrB_Row_assign);
implement_insert_vector_into_row_trait!(i8, i8, GrB_Row_assign);
implement_insert_vector_into_row_trait!(i16, i16, GrB_Row_assign);
implement_insert_vector_into_row_trait!(i32, i32, GrB_Row_assign);
implement_insert_vector_into_row_trait!(i64, i64, GrB_Row_assign);
implement_insert_vector_into_row_trait!(f32, f32, GrB_Row_assign);
implement_insert_vector_into_row_trait!(f64, f64, GrB_Row_assign);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;

    use crate::value_types::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };
    use crate::value_types::sparse_vector::{FromVectorElementList, VectorElementList};

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
            &First::<u8, u8, u8>::new(),
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
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mask_element_list = VectorElementList::<bool>::from_element_vector(vec![
            (1, false).into(),
            (2, true).into(),
            (4, true).into(),
            // (5, true).into(),
        ]);
        let mask = SparseVector::<bool>::from_element_list(
            &context,
            &matrix_size.column_width(),
            &mask_element_list,
            &First::<bool, bool, bool>::new(),
        )
        .unwrap();

        let indices_to_insert: Vec<ElementIndex> = (0..vector_to_insert_length).collect();
        let indices_to_insert = ElementIndexSelector::Index(&indices_to_insert);

        let insert_operator = InsertVectorIntoRow::new(&OperatorOptions::new_default(), None);

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
        assert_eq!(matrix.get_element_value(&(2, 4).into()).unwrap(), 11);

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
                &indices_to_insert,
                &row_to_insert_into,
                &vector_to_insert,
                &mask.into(),
            )
            .unwrap();

        println!("{}", matrix);

        assert_eq!(matrix.number_of_stored_elements().unwrap(), 5);
        assert_eq!(matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
        assert_eq!(matrix.get_element_value(&(2, 2).into()).unwrap(), 3);
        assert_eq!(matrix.get_element_value(&(1, 2).into()).unwrap(), 1);
        assert_eq!(matrix.get_element_value(&(5, 2).into()).unwrap(), 12);
        assert_eq!(matrix.get_element_value(&(2, 4).into()).unwrap(), 11);
        assert_eq!(matrix.get_element_value(&(2, 1).into()).unwrap(), 0);
    }
}
