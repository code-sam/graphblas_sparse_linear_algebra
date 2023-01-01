use std::marker::PhantomData;
use std::ptr;

use suitesparse_graphblas_sys::GrB_IndexUnaryOp;

use crate::collections::sparse_matrix::{GraphblasSparseMatrixTrait, SparseMatrix};
use crate::collections::sparse_scalar::{GraphblasSparseScalarTrait, SparseScalar};
use crate::collections::sparse_vector::{GraphblasSparseVectorTrait, SparseVector};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::apply::IndexUnaryOperatorApplier;
use crate::operators::binary_operator::Second;
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::operators::{
    binary_operator::BinaryOperator, options::OperatorOptions, unary_operator::UnaryOperator,
};
use crate::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_apply_IndexOp_Scalar, GrB_UnaryOp,
    GrB_Vector_apply_IndexOp_Scalar,
};

pub trait ApplyIndexUnaryOperatorWithSparseScalar<
    FirstArgument,
    SecondArgument,
    Product,
    EvaluationDomain,
> where
    FirstArgument: ValueType,
    SecondArgument: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
{
    fn apply_to_vector(
        &self,
        vector: &SparseVector<FirstArgument>,
        argument: &SparseScalar<SecondArgument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_vector_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        vector: &SparseVector<FirstArgument>,
        argument: &SparseScalar<SecondArgument>,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix(
        &self,
        matrix: &SparseMatrix<FirstArgument>,
        argument: &SparseScalar<SecondArgument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        matrix: &SparseMatrix<FirstArgument>,
        argument: &SparseScalar<SecondArgument>,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<
        FirstArgument: ValueType,
        SecondArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    >
    ApplyIndexUnaryOperatorWithSparseScalar<
        FirstArgument,
        SecondArgument,
        Product,
        EvaluationDomain,
    > for IndexUnaryOperatorApplier<FirstArgument, SecondArgument, Product, EvaluationDomain>
{
    fn apply_to_vector(
        &self,
        vector: &SparseVector<FirstArgument>,
        argument: &SparseScalar<SecondArgument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context();

        context.call(
            || unsafe {
                GrB_Vector_apply_IndexOp_Scalar(
                    product.graphblas_vector(),
                    ptr::null_mut(),
                    self.accumulator(),
                    self.index_unary_operator(),
                    vector.graphblas_vector(),
                    argument.graphblas_scalar(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_to_vector_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        vector: &SparseVector<FirstArgument>,
        argument: &SparseScalar<SecondArgument>,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context();

        context.call(
            || unsafe {
                GrB_Vector_apply_IndexOp_Scalar(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    self.accumulator(),
                    self.index_unary_operator(),
                    vector.graphblas_vector(),
                    argument.graphblas_scalar(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_to_matrix(
        &self,
        matrix: &SparseMatrix<FirstArgument>,
        argument: &SparseScalar<SecondArgument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_IndexOp_Scalar(
                    product.graphblas_matrix(),
                    ptr::null_mut(),
                    self.accumulator(),
                    self.index_unary_operator(),
                    matrix.graphblas_matrix(),
                    argument.graphblas_scalar(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;

        Ok(())
    }

    fn apply_to_matrix_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        matrix: &SparseMatrix<FirstArgument>,
        argument: &SparseScalar<SecondArgument>,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = argument.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_IndexOp_Scalar(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    self.accumulator(),
                    self.index_unary_operator(),
                    matrix.graphblas_matrix(),
                    argument.graphblas_scalar(),
                    self.options(),
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
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };
    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, VectorElementList,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::operators::index_unary_operator::IsValueGreaterThan;

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
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<f32>::new(&context, &matrix_size).unwrap();

        let operator = IndexUnaryOperatorApplier::new(
            &IsValueGreaterThan::<u8, i8, f32, i16>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        operator
            .apply_to_matrix(&matrix, &argument, &mut product_matrix)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix.get_element_value(&(1, 1).into()).unwrap(),
            0.0
        );
        assert_eq!(
            product_matrix.get_element_value(&(2, 1).into()).unwrap(),
            0.0
        );
        assert_eq!(
            product_matrix.get_element_value(&(4, 2).into()).unwrap(),
            1.0
        );
        assert_eq!(
            product_matrix.get_element_value(&(5, 2).into()).unwrap(),
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
    //         &context.clone(),
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
