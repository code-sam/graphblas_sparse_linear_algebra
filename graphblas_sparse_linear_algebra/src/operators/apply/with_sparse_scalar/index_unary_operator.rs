use std::ptr;

use crate::collections::sparse_matrix::GraphblasSparseMatrixTrait;
use crate::collections::sparse_scalar::GraphblasSparseScalarTrait;
use crate::collections::sparse_vector::GraphblasSparseVectorTrait;
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::apply::IndexUnaryOperatorApplier;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::operators::mask::{MatrixMask, VectorMask};
use crate::operators::options::{OperatorOptions, OperatorOptionsTrait};
use crate::value_type::ValueType;

use crate::bindings_to_graphblas_implementation::{
    GrB_Matrix_apply_IndexOp_Scalar, GrB_Vector_apply_IndexOp_Scalar,
};

pub trait ApplyIndexUnaryOperatorWithSparseScalar<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn apply_to_vector(
        &self,
        vector: &(impl GraphblasSparseVectorTrait + ContextTrait),
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseScalarTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl VectorMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix(
        &self,
        matrix: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseScalarTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        mask: &(impl MatrixMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyIndexUnaryOperatorWithSparseScalar<EvaluationDomain>
    for IndexUnaryOperatorApplier
{
    fn apply_to_vector(
        &self,
        vector: &(impl GraphblasSparseVectorTrait + ContextTrait),
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseScalarTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        mask: &(impl VectorMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context();

        context.call(
            || unsafe {
                GrB_Vector_apply_IndexOp_Scalar(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    vector.graphblas_vector(),
                    argument.graphblas_scalar(),
                    options.to_graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_to_matrix(
        &self,
        matrix: &(impl GraphblasSparseMatrixTrait + ContextTrait),
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &(impl GraphblasSparseScalarTrait + ContextTrait),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GraphblasSparseMatrixTrait + ContextTrait),
        mask: &(impl MatrixMask + ContextTrait),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_IndexOp_Scalar(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    matrix.graphblas_matrix(),
                    argument.graphblas_scalar(),
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
    use crate::collections::sparse_scalar::SparseScalar;
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::index_unary_operator::IsValueGreaterThan;
    use crate::operators::mask::SelectEntireMatrix;
    use crate::operators::options::OperatorOptions;

    #[test]
    fn test_matrix_index_unary_operator() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 1, 2).into(),
            (4, 2, 4).into(),
            (5, 2, 5).into(),
        ]);

        let argument = SparseScalar::<i8>::from_value(&context, 2i8).unwrap();

        let matrix_size: Size = (10, 15).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<f32>::new(&context, &matrix_size).unwrap();

        let operator = IndexUnaryOperatorApplier::new();

        operator
            .apply_to_matrix(
                &matrix,
                &IsValueGreaterThan::<i16>::new(),
                &argument,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            0.0
        );
        assert_eq!(
            product_matrix
                .get_element_value(&(2, 1).into())
                .unwrap()
                .unwrap(),
            0.0
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(4, 2).into())
                .unwrap(),
            1.0
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(5, 2).into())
                .unwrap(),
            1.0
        );
    }

    // #[test]
    // fn test_vector_unary_operator() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();

    //     let element_list = VectorElementList::<u8>::from_element_vector(vec![
    //         (1, 1).into(),
    //         (2, 2).into(),
    //         (4, 4).into(),
    //         (5, 5).into(),
    //     ]);

    //     let vector_length: usize = 10;
    //     let vector = SparseVector::<u8>::from_element_list(
    //         &context.to_owned(),
    //         &vector_length,
    //         &element_list,
    //         &First::<u8, u8, u8, u8>::new(),
    //     )
    //     .unwrap();

    //     let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

    //     let operator = UnaryOperatorApplier::new(
    //         &One::<u8, u8, u8>::new(),
    //         &OperatorOptions::new_default(),
    //         None,
    //     );

    //     operator
    //         .apply_to_vector(&vector, &mut product_vector)
    //         .unwrap();

    //     println!("{}", product_vector);

    //     assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
    //     assert_eq!(product_vector.get_element_value(&2).unwrap(), 1);
    //     assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

    //     let operator = UnaryOperatorApplier::new(
    //         &Identity::<u8, u8, u8>::new(),
    //         &OperatorOptions::new_default(),
    //         None,
    //     );
    //     operator
    //         .apply_to_vector(&vector, &mut product_vector)
    //         .unwrap();

    //     println!("{}", vector);
    //     println!("{}", product_vector);

    //     assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
    //     assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
    //     assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    // }

    // #[test]
    // fn test_vector_unary_negation_operator() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();

    //     let vector_length: usize = 10;
    //     let vector = SparseVector::<bool>::new(&context, &vector_length).unwrap();

    //     let mut product_vector = SparseVector::<bool>::new(&context, &vector_length).unwrap();

    //     let operator = UnaryOperatorApplier::new(
    //         &LogicalNegation::<bool, bool, bool>::new(),
    //         &OperatorOptions::new_default(),
    //         None,
    //     );

    //     operator
    //         .apply_to_vector(&vector, &mut product_vector)
    //         .unwrap();

    //     println!("{}", product_vector);

    //     assert_eq!(product_vector.number_of_stored_elements().unwrap(), 0);
    // }
}
