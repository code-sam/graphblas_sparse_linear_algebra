use std::ptr;

use crate::collections::sparse_matrix::GraphblasSparseMatrixTrait;
use crate::collections::sparse_vector::GraphblasSparseVectorTrait;
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::{MatrixMask, VectorMask};
use crate::operators::options::OperatorOptionsTrait;
use crate::operators::{options::OperatorOptions, unary_operator::UnaryOperator};
use crate::value_type::ValueType;

use crate::bindings_to_graphblas_implementation::{GrB_Matrix_apply, GrB_Vector_apply};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for UnaryOperatorApplier {}
unsafe impl Sync for UnaryOperatorApplier {}

#[derive(Debug, Clone)]
pub struct UnaryOperatorApplier {}

impl UnaryOperatorApplier {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyUnaryOperator<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn apply_to_vector(
        &self,
        operator: &impl UnaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl VectorMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix(
        &self,
        operator: &impl UnaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        mask: &(impl MatrixMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyUnaryOperator<EvaluationDomain> for UnaryOperatorApplier {
    fn apply_to_vector(
        &self,
        operator: &impl UnaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl VectorMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context();

        context.call(
            || unsafe {
                GrB_Vector_apply(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    argument.graphblas_vector(),
                    options.to_graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_to_matrix(
        &self,
        operator: &impl UnaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        mask: &(impl MatrixMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    argument.graphblas_matrix(),
                    options.to_graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size, SparseMatrix,
    };
    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, SparseVector, VectorElementList,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::mask::{SelectEntireMatrix, SelectEntireVector};
    use crate::operators::unary_operator::{Identity, LogicalNegation, One};

    #[test]
    fn test_matrix_unary_operator() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 1, 2).into(),
            (4, 2, 4).into(),
            (5, 2, 5).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let operator = UnaryOperatorApplier::new();

        operator
            .apply_to_matrix(
                &One::<u8>::new(),
                &matrix,
                &Assignment::<u8>::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(2, 1).into())
                .unwrap(),
            1
        );
        assert_eq!(
            product_matrix.get_element_value(&(9, 1).into()).unwrap(),
            None
        );

        let operator = UnaryOperatorApplier::new();
        operator
            .apply_to_matrix(
                &Identity::<u8>::new(),
                &matrix,
                &Assignment::<u8>::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(2, 1).into())
                .unwrap(),
            2
        );
        assert_eq!(
            product_matrix.get_element_value(&(9, 1).into()).unwrap(),
            None
        );
    }

    #[test]
    fn test_vector_unary_operator() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator = UnaryOperatorApplier::new();

        operator
            .apply_to_vector(
                &One::<u8>::new(),
                &vector,
                &Assignment::<u8>::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 1);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);

        let operator = UnaryOperatorApplier::new();
        operator
            .apply_to_vector(
                &Identity::<u8>::new(),
                &vector,
                &Assignment::<u8>::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), None);
    }

    #[test]
    fn test_vector_unary_negation_operator() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let vector_length: usize = 10;
        let vector = SparseVector::<bool>::new(&context, &vector_length).unwrap();

        let mut product_vector = SparseVector::<bool>::new(&context, &vector_length).unwrap();

        let operator = UnaryOperatorApplier::new();

        operator
            .apply_to_vector(
                &LogicalNegation::<bool>::new(),
                &vector,
                &Assignment::<bool>::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 0);
    }
}
