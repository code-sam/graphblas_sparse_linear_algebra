use std::marker::PhantomData;
use std::ptr;

use crate::bindings_to_graphblas_implementation::{GrB_BinaryOp, GrB_Descriptor, GrB_transpose};
use crate::collections::sparse_matrix::{GraphblasSparseMatrixTrait, SparseMatrix};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::{AsBoolean, ValueType};

use super::binary_operator::AccumulatorBinaryOperator;

#[derive(Debug, Clone)]
pub struct MatrixTranspose<Product>
where
    Product: ValueType,
{
    _product: PhantomData<Product>,

    accumulator: GrB_BinaryOp,
    options: GrB_Descriptor,
}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<Product: ValueType> Send for MatrixTranspose<Product> {}
unsafe impl<Product: ValueType> Sync for MatrixTranspose<Product> {}

impl<Product> MatrixTranspose<Product>
where
    Product: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<Product>,
    ) -> Self {
        Self {
            accumulator: accumulator.accumulator_graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _product: PhantomData,
        }
    }
}

pub trait TransposeMatrix<Product: ValueType> {
    fn apply(
        &self,
        matrix: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        transpose: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_mask(
        &self,
        matrix: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        transpose: &mut SparseMatrix<Product>,
        mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<Product: ValueType> TransposeMatrix<Product> for MatrixTranspose<Product> {
    fn apply(
        &self,
        matrix: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        transpose: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = transpose.context();

        context.call(
            || unsafe {
                GrB_transpose(
                    transpose.graphblas_matrix(),
                    ptr::null_mut(),
                    self.accumulator,
                    matrix.graphblas_matrix(),
                    self.options,
                )
            },
            unsafe { transpose.graphblas_matrix_ref() },
        )?;

        Ok(())
    }

    fn apply_with_mask(
        &self,
        matrix: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        transpose: &mut SparseMatrix<Product>,
        mask: &(impl GraphblasSparseMatrixTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = transpose.context();

        context.call(
            || unsafe {
                GrB_transpose(
                    transpose.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    self.accumulator,
                    matrix.graphblas_matrix(),
                    self.options,
                )
            },
            unsafe { transpose.graphblas_matrix_ref() },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList,
    };
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};

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
            &First::<u8>::new(),
        )
        .unwrap();

        let mut matrix_transpose = SparseMatrix::<u8>::new(&context, &(2, 2).into()).unwrap();

        let transpose_operator =
            MatrixTranspose::new(&OperatorOptions::new_default(), &Assignment::<u8>::new());

        transpose_operator
            .apply(&matrix, &mut matrix_transpose)
            .unwrap();

        assert_eq!(
            matrix_transpose
                .get_element_value_or_default(&(0, 0).into())
                .unwrap(),
            1
        );
        assert_eq!(
            matrix_transpose
                .get_element_value_or_default(&(1, 0).into())
                .unwrap(),
            3
        );
        assert_eq!(
            matrix_transpose
                .get_element_value_or_default(&(0, 1).into())
                .unwrap(),
            2
        );
        assert_eq!(
            matrix_transpose
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            4
        );
    }
}
