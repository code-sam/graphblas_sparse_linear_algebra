use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_scalar::GetGraphblasSparseScalar;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::CallGraphBlasContext;
use crate::error::SparseLinearAlgebraError;
use crate::operators::apply::IndexUnaryOperatorApplier;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::operators::mask::{MatrixMask, VectorMask};
use crate::operators::options::GetOperatorOptions;
use crate::value_type::ValueType;

use crate::graphblas_bindings::{GrB_Matrix_apply_IndexOp_Scalar, GrB_Vector_apply_IndexOp_Scalar};

pub trait ApplyIndexUnaryOperatorWithSparseScalar<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn apply_to_vector(
        &self,
        vector: &impl GetGraphblasSparseVector,
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &impl GetGraphblasSparseScalar,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix(
        &self,
        matrix: &impl GetGraphblasSparseMatrix,
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &impl GetGraphblasSparseScalar,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> ApplyIndexUnaryOperatorWithSparseScalar<EvaluationDomain>
    for IndexUnaryOperatorApplier
{
    fn apply_to_vector(
        &self,
        vector: &impl GetGraphblasSparseVector,
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &impl GetGraphblasSparseScalar,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseVector,
        mask: &impl VectorMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context_ref();

        context.call(
            || unsafe {
                GrB_Vector_apply_IndexOp_Scalar(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    vector.graphblas_vector(),
                    argument.graphblas_scalar(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_to_matrix(
        &self,
        matrix: &impl GetGraphblasSparseMatrix,
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &impl GetGraphblasSparseScalar,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut impl GetGraphblasSparseMatrix,
        mask: &impl MatrixMask,
        options: &impl GetOperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context_ref();

        context.call(
            || unsafe {
                GrB_Matrix_apply_IndexOp_Scalar(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    matrix.graphblas_matrix(),
                    argument.graphblas_scalar(),
                    options.graphblas_descriptor(),
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

    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementValue,
    };
    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::collections::sparse_scalar::SparseScalar;
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::index_unary_operator::IsValueGreaterThan;
    use crate::operators::mask::SelectEntireMatrix;
    use crate::operators::options::OperatorOptions;

    #[test]
    fn test_matrix_index_unary_operator() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 1, 2).into(),
            (4, 2, 4).into(),
            (5, 2, 5).into(),
        ]);

        let argument = SparseScalar::<i8>::from_value(context.clone(), 2i8).unwrap();

        let matrix_size: Size = (10, 15).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            matrix_size,
            element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<f32>::new(context.clone(), matrix_size).unwrap();

        let operator = IndexUnaryOperatorApplier::new();

        operator
            .apply_to_matrix(
                &matrix,
                &IsValueGreaterThan::<i16>::new(),
                &argument,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(context.clone()),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.element_value_or_default(1, 1).unwrap(), 0.0);
        assert_eq!(product_matrix.element_value(2, 1).unwrap().unwrap(), 0.0);
        assert_eq!(product_matrix.element_value_or_default(4, 2).unwrap(), 1.0);
        assert_eq!(product_matrix.element_value_or_default(5, 2).unwrap(), 1.0);
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
