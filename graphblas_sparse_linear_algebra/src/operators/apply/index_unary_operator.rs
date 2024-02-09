use suitesparse_graphblas_sys::{
    GrB_Matrix_apply_IndexOp_BOOL, GrB_Matrix_apply_IndexOp_FP32, GrB_Matrix_apply_IndexOp_FP64,
    GrB_Matrix_apply_IndexOp_INT16, GrB_Matrix_apply_IndexOp_INT32, GrB_Matrix_apply_IndexOp_INT64,
    GrB_Matrix_apply_IndexOp_INT8, GrB_Matrix_apply_IndexOp_UINT16,
    GrB_Matrix_apply_IndexOp_UINT32, GrB_Matrix_apply_IndexOp_UINT64,
    GrB_Matrix_apply_IndexOp_UINT8, GrB_Vector_apply_IndexOp_BOOL, GrB_Vector_apply_IndexOp_FP32,
    GrB_Vector_apply_IndexOp_FP64, GrB_Vector_apply_IndexOp_INT16, GrB_Vector_apply_IndexOp_INT32,
    GrB_Vector_apply_IndexOp_INT64, GrB_Vector_apply_IndexOp_INT8, GrB_Vector_apply_IndexOp_UINT16,
    GrB_Vector_apply_IndexOp_UINT32, GrB_Vector_apply_IndexOp_UINT64,
    GrB_Vector_apply_IndexOp_UINT8,
};

use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::operators::mask::{MatrixMask, VectorMask};
use crate::operators::options::GetGraphblasDescriptor;

use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_2_typed_graphblas_functions_with_implementation_type;
use crate::value_type::{ConvertScalar, ValueType};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for IndexUnaryOperatorApplier {}
unsafe impl Sync for IndexUnaryOperatorApplier {}

#[derive(Debug, Clone)]
pub struct IndexUnaryOperatorApplier {}

impl IndexUnaryOperatorApplier {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait ApplyIndexUnaryOperator<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn apply_to_vector(
        &self,
        vector: &(impl GetGraphblasSparseVector + GetContext),
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &EvaluationDomain,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix(
        &self,
        matrix: &(impl GetGraphblasSparseMatrix + GetContext),
        operator: &impl IndexUnaryOperator<EvaluationDomain>,
        argument: &EvaluationDomain,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseMatrix + GetContext),
        mask: &(impl MatrixMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_apply_index_binary_operator {
    ($evaluation_domain: ty, $_implementation_type: ty, $graphblas_function_1: ident, $graphblas_function_2: ident) => {
        impl ApplyIndexUnaryOperator<$evaluation_domain> for IndexUnaryOperatorApplier {
            fn apply_to_vector(
                &self,
                vector: &(impl GetGraphblasSparseVector + GetContext),
                operator: &impl IndexUnaryOperator<$evaluation_domain>,
                argument: &$evaluation_domain,
                accumulator: &impl AccumulatorBinaryOperator<$evaluation_domain>,
                product: &mut (impl GetGraphblasSparseVector + GetContext),
                mask: &(impl VectorMask + GetContext),
                options: &impl GetGraphblasDescriptor,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let argument = argument.to_owned().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_1(
                            product.graphblas_vector(),
                            mask.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            vector.graphblas_vector(),
                            argument,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_to_matrix(
                &self,
                matrix: &(impl GetGraphblasSparseMatrix + GetContext),
                operator: &impl IndexUnaryOperator<$evaluation_domain>,
                argument: &$evaluation_domain,
                accumulator: &impl AccumulatorBinaryOperator<$evaluation_domain>,
                product: &mut (impl GetGraphblasSparseMatrix + GetContext),
                mask: &(impl MatrixMask + GetContext),
                options: &impl GetGraphblasDescriptor,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let argument = argument.to_owned().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_2(
                            product.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            accumulator.accumulator_graphblas_type(),
                            operator.graphblas_type(),
                            matrix.graphblas_matrix(),
                            argument,
                            options.graphblas_descriptor(),
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_2_typed_graphblas_functions_with_implementation_type!(
    implement_apply_index_binary_operator,
    GrB_Vector_apply_IndexOp,
    GrB_Matrix_apply_IndexOp
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::{
        FromMatrixElementList, GetSparseMatrixElementValue,
    };
    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
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

        let argument = 2i8;

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
                &IsValueGreaterThan::<i8>::new(),
                &argument,
                &Assignment::<i8>::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.element_value(&1, &1).unwrap(), Some(0.0));
        assert_eq!(product_matrix.element_value(&2, &1).unwrap(), Some(0.0));
        assert_eq!(
            product_matrix
                .element_value(&4, &2)
                .unwrap()
                .unwrap_or_default(),
            1.0
        );
        assert_eq!(
            product_matrix.element_value_or_default(&5, &2).unwrap(),
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
