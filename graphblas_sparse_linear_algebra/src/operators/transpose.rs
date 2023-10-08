use std::ptr;

use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::graphblas_bindings::GrB_transpose;
use crate::operators::options::OperatorOptions;
use crate::value_type::ValueType;

use super::binary_operator::AccumulatorBinaryOperator;
use super::mask::MatrixMask;
use super::options::OperatorOptionsTrait;

#[derive(Debug, Clone)]
pub struct MatrixTranspose {}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for MatrixTranspose {}
unsafe impl Sync for MatrixTranspose {}

impl MatrixTranspose {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait TransposeMatrix<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        matrix: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        transpose: &mut (impl GetGraphblasSparseMatrix + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_mask(
        &self,
        matrix: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        transpose: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> TransposeMatrix<EvaluationDomain> for MatrixTranspose {
    fn apply(
        &self,
        matrix: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        transpose: &mut (impl GetGraphblasSparseMatrix + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = transpose.context();

        context.call(
            || unsafe {
                GrB_transpose(
                    transpose.graphblas_matrix(),
                    ptr::null_mut(),
                    accumulator.accumulator_graphblas_type(),
                    matrix.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
                )
            },
            unsafe { transpose.graphblas_matrix_ref() },
        )?;

        Ok(())
    }

    fn apply_with_mask(
        &self,
        matrix: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        transpose: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = transpose.context();

        context.call(
            || unsafe {
                GrB_transpose(
                    transpose.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    matrix.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
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
    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetMatrixElementValue,
    };
    use crate::collections::sparse_matrix::{MatrixElementList, SparseMatrix};
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::SelectEntireMatrix;

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

        let transpose_operator = MatrixTranspose::new();

        transpose_operator
            .apply_with_mask(
                &matrix,
                &Assignment::<u8>::new(),
                &mut matrix_transpose,
                &SelectEntireMatrix::new(&context),
                &OperatorOptions::new_default(),
            )
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
