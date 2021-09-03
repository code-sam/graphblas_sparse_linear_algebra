use std::marker::PhantomData;
use std::ptr;

use crate::bindings_to_graphblas_implementation::{GrB_BinaryOp, GrB_Descriptor, GrB_transpose};
use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator, mask::MatrixMask, options::OperatorOptions,
};
use crate::value_types::sparse_matrix::SparseMatrix;
use crate::value_types::value_type::{AsBoolean, ValueType};

#[derive(Debug, Clone)]
pub struct MatrixTranspose<Applicant, Product>
where
    Applicant: ValueType,
    Product: ValueType,
{
    _applicant: PhantomData<Applicant>,
    _product: PhantomData<Product>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for MatrixTranspose<bool,bool> {}
unsafe impl Send for MatrixTranspose<u8,u8> {}
unsafe impl Send for MatrixTranspose<u16,u16> {}
unsafe impl Send for MatrixTranspose<u32,u32> {}
unsafe impl Send for MatrixTranspose<u64,u64> {}
unsafe impl Send for MatrixTranspose<i8,i8> {}
unsafe impl Send for MatrixTranspose<i16,i16> {}
unsafe impl Send for MatrixTranspose<i32,i32> {}
unsafe impl Send for MatrixTranspose<i64,i64> {}
unsafe impl Send for MatrixTranspose<f32,f32> {}
unsafe impl Send for MatrixTranspose<f64,f64> {}

unsafe impl Sync for MatrixTranspose<bool,bool> {}
unsafe impl Sync for MatrixTranspose<u8,u8> {}
unsafe impl Sync for MatrixTranspose<u16,u16> {}
unsafe impl Sync for MatrixTranspose<u32,u32> {}
unsafe impl Sync for MatrixTranspose<u64,u64> {}
unsafe impl Sync for MatrixTranspose<i8,i8> {}
unsafe impl Sync for MatrixTranspose<i16,i16> {}
unsafe impl Sync for MatrixTranspose<i32,i32> {}
unsafe impl Sync for MatrixTranspose<i64,i64> {}
unsafe impl Sync for MatrixTranspose<f32,f32> {}
unsafe impl Sync for MatrixTranspose<f64,f64> {}

impl<Applicant, Product> MatrixTranspose<Applicant, Product>
where
    Applicant: ValueType,
    Product: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<Product, Product, Product>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _applicant: PhantomData,
            _product: PhantomData,
        }
    }

    pub fn apply(
        &self,
        matrix: &SparseMatrix<Applicant>,
        transpose: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = transpose.context();

        let transpose_with_write_lock = transpose.get_write_lock()?;
        let matrix_with_read_lock = matrix.get_read_lock()?;
        context.call(|| unsafe {
            GrB_transpose(
                *transpose_with_write_lock,
                ptr::null_mut(),
                self.accumulator,
                *matrix_with_read_lock,
                self.options,
            )
        })?;

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        matrix: &SparseMatrix<Applicant>,
        transpose: &mut SparseMatrix<Product>,
        mask: &MatrixMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = transpose.context();

        let transpose_with_write_lock = transpose.get_write_lock()?;
        let mask_with_read_lock = mask.get_read_lock()?;
        let matrix_with_read_lock = matrix.get_read_lock()?;
        context.call(|| unsafe {
            GrB_transpose(
                *transpose_with_write_lock,
                *mask_with_read_lock,
                self.accumulator,
                *matrix_with_read_lock,
                self.options,
            )
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::value_types::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList,
    };

    #[test]
    fn test_transpose() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &(2, 2).into(),
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut matrix_transpose = SparseMatrix::<u8>::new(&context, &(2, 2).into()).unwrap();

        let transpose_operator = MatrixTranspose::new(&OperatorOptions::new_default(), None);

        transpose_operator
            .apply(&matrix, &mut matrix_transpose)
            .unwrap();

        assert_eq!(
            matrix_transpose.get_element_value(&(0, 0).into()).unwrap(),
            1
        );
        assert_eq!(
            matrix_transpose.get_element_value(&(1, 0).into()).unwrap(),
            3
        );
        assert_eq!(
            matrix_transpose.get_element_value(&(0, 1).into()).unwrap(),
            2
        );
        assert_eq!(
            matrix_transpose.get_element_value(&(1, 1).into()).unwrap(),
            4
        );
    }
}
