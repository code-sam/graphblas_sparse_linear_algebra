use std::ptr;

use suitesparse_graphblas_sys::{
    GrB_Matrix_apply_BinaryOp1st_Scalar, GrB_Matrix_apply_BinaryOp2nd_Scalar,
    GrB_Vector_apply_BinaryOp1st_Scalar, GrB_Vector_apply_BinaryOp2nd_Scalar,
};

use crate::collections::sparse_matrix::{GraphblasSparseMatrixTrait, SparseMatrix};
use crate::collections::sparse_scalar::{GraphblasSparseScalarTrait, SparseScalar};
use crate::collections::sparse_vector::{GraphblasSparseVectorTrait, SparseVector};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::apply::BinaryOperatorApplier;
use crate::value_type::{AsBoolean, ValueType};

pub trait ApplyBinaryOperatorWithSparseScalar<
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
    fn apply_with_vector_as_first_argument(
        &self,
        first_argument: &SparseVector<FirstArgument>,
        second_argument: &SparseScalar<SecondArgument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_second_argument(
        &self,
        first_argument: &SparseScalar<FirstArgument>,
        second_argument: &SparseVector<SecondArgument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_first_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseVector<FirstArgument>,
        second_argument: &SparseScalar<SecondArgument>,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_second_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseScalar<FirstArgument>,
        second_argument: &SparseVector<SecondArgument>,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_first_argument(
        &self,
        first_argument: &SparseMatrix<FirstArgument>,
        second_argument: &SparseScalar<SecondArgument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_second_argument(
        &self,
        first_argument: &SparseScalar<FirstArgument>,
        second_argument: &SparseMatrix<SecondArgument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_first_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseMatrix<FirstArgument>,
        second_argument: &SparseScalar<SecondArgument>,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_second_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseScalar<FirstArgument>,
        second_argument: &SparseMatrix<SecondArgument>,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<
        FirstArgument: ValueType,
        SecondArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > ApplyBinaryOperatorWithSparseScalar<FirstArgument, SecondArgument, Product, EvaluationDomain>
    for BinaryOperatorApplier<FirstArgument, SecondArgument, Product, EvaluationDomain>
{
    fn apply_with_vector_as_first_argument(
        &self,
        first_argument: &SparseVector<FirstArgument>,
        second_argument: &SparseScalar<SecondArgument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();
        let second_argument = second_argument.clone();

        context.call(
            || unsafe {
                GrB_Vector_apply_BinaryOp2nd_Scalar(
                    product.graphblas_vector(),
                    ptr::null_mut(),
                    self.accumulator(),
                    self.binary_operator(),
                    first_argument.graphblas_vector(),
                    second_argument.graphblas_scalar(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_with_vector_as_second_argument(
        &self,
        first_argument: &SparseScalar<FirstArgument>,
        second_argument: &SparseVector<SecondArgument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Vector_apply_BinaryOp1st_Scalar(
                    product.graphblas_vector(),
                    ptr::null_mut(),
                    self.accumulator(),
                    self.binary_operator(),
                    first_argument.graphblas_scalar(),
                    second_argument.graphblas_vector(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_with_vector_as_first_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseVector<FirstArgument>,
        second_argument: &SparseScalar<SecondArgument>,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Vector_apply_BinaryOp2nd_Scalar(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    self.accumulator(),
                    self.binary_operator(),
                    first_argument.graphblas_vector(),
                    second_argument.graphblas_scalar(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_with_vector_as_second_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseScalar<FirstArgument>,
        second_argument: &SparseVector<SecondArgument>,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Vector_apply_BinaryOp1st_Scalar(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    self.accumulator(),
                    self.binary_operator(),
                    first_argument.graphblas_scalar(),
                    second_argument.graphblas_vector(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_vector() },
        )?;

        Ok(())
    }

    fn apply_with_matrix_as_first_argument(
        &self,
        first_argument: &SparseMatrix<FirstArgument>,
        second_argument: &SparseScalar<SecondArgument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_BinaryOp2nd_Scalar(
                    product.graphblas_matrix(),
                    ptr::null_mut(),
                    self.accumulator(),
                    self.binary_operator(),
                    first_argument.graphblas_matrix(),
                    second_argument.graphblas_scalar(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;

        Ok(())
    }

    fn apply_with_matrix_as_second_argument(
        &self,
        first_argument: &SparseScalar<FirstArgument>,
        second_argument: &SparseMatrix<SecondArgument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_BinaryOp1st_Scalar(
                    product.graphblas_matrix(),
                    ptr::null_mut(),
                    self.accumulator(),
                    self.binary_operator(),
                    first_argument.graphblas_scalar(),
                    second_argument.graphblas_matrix(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;

        Ok(())
    }

    fn apply_with_matrix_as_first_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseMatrix<FirstArgument>,
        second_argument: &SparseScalar<SecondArgument>,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_BinaryOp2nd_Scalar(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    self.accumulator(),
                    self.binary_operator(),
                    first_argument.graphblas_matrix(),
                    second_argument.graphblas_scalar(),
                    self.options(),
                )
            },
            unsafe { &product.graphblas_matrix() },
        )?;

        Ok(())
    }

    fn apply_with_matrix_as_second_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseScalar<FirstArgument>,
        second_argument: &SparseMatrix<SecondArgument>,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_apply_BinaryOp1st_Scalar(
                    product.graphblas_matrix(),
                    mask.graphblas_matrix(),
                    self.accumulator(),
                    self.binary_operator(),
                    first_argument.graphblas_scalar(),
                    second_argument.graphblas_matrix(),
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
    use crate::operators::binary_operator::{First, Plus};
    use crate::operators::options::OperatorOptions;

    #[test]
    fn test_matrix_binary_operator_application() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 1, 2).into(),
            (4, 2, 4).into(),
            (5, 2, 5).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let operator = BinaryOperatorApplier::new(
            &First::<u8, u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        let second_agrument = SparseScalar::<u8>::from_value(&context, 10).unwrap();
        operator
            .apply_with_matrix_as_first_argument(&matrix, &second_agrument, &mut product_matrix)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.get_element_value(&(2, 1).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(9, 1).into()).unwrap(), 0);

        let operator = BinaryOperatorApplier::new(
            &First::<u8, u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );
        let first_agrument = SparseScalar::<u8>::from_value(&context, 10).unwrap();
        operator
            .apply_with_matrix_as_second_argument(&first_agrument, &matrix, &mut product_matrix)
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix.get_element_value(&(2, 1).into()).unwrap(),
            10
        );
        assert_eq!(product_matrix.get_element_value(&(9, 1).into()).unwrap(), 0);
    }

    #[test]
    fn test_vector_binary_operator_application() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new(
            &First::<u8, u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        let second_agrument = SparseScalar::<u8>::from_value(&context, 10).unwrap();
        operator
            .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        let operator = BinaryOperatorApplier::new(
            &First::<u8, u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );
        let first_argument = SparseScalar::<u8>::from_value(&context, 10).unwrap();
        operator
            .apply_with_vector_as_second_argument(&first_argument, &vector, &mut product_vector)
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 10);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    }

    #[test]
    fn test_vector_binary_operator_application_with_usize() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<usize>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<usize>::from_element_list(
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<usize, usize, usize, usize>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<usize>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new(
            &First::<usize, usize, usize, usize>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        let second_agrument = SparseScalar::<usize>::from_value(&context, 10).unwrap();
        operator
            .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        let operator = BinaryOperatorApplier::new(
            &First::<usize, usize, usize, usize>::new(),
            &OperatorOptions::new_default(),
            None,
        );
        let first_agrument = SparseScalar::<usize>::from_value(&context, 10).unwrap();
        operator
            .apply_with_vector_as_second_argument(&first_agrument, &vector, &mut product_vector)
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 10);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    }

    #[test]
    fn test_type_casting() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<i8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new(
            &Plus::<u8, bool, i8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        let second_agrument = SparseScalar::<bool>::from_value(&context, true).unwrap();
        operator
            .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 3);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        let operator = BinaryOperatorApplier::new(
            &Plus::<u8, bool, i8, bool>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        operator
            .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 1);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        // let operator = BinaryOperatorApplier::new(
        //     &First::<u8, u8, u8, u8>::new(),
        //     &OperatorOptions::new_default(),
        //     None,
        // );
        // let first_argument = SparseScalar::<u8>::from_value(&context, 10).unwrap();
        // operator
        //     .apply_with_vector_as_second_argument(&first_argument, &vector, &mut product_vector)
        //     .unwrap();

        // println!("{}", vector);
        // println!("{}", product_vector);

        // assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        // assert_eq!(product_vector.get_element_value(&2).unwrap(), 10);
        // assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    }

    // #[test]
    // fn test_operator_destructor() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();
    //     // Test if this causes a memory leak, due to the absence of an explicit call to GrB_free.
    //     for i in 0..(1e5 as usize) {
    //         let element_list = VectorElementList::<usize>::from_element_vector(vec![
    //             (1, 1).into(),
    //             (2, 2).into(),
    //             (4, 4).into(),
    //             (5, 5).into(),
    //             (10+i, i).into(),
    //         ]);

    //         let vector_length: usize = 100+i;
    //         let vector = SparseVector::<usize>::from_element_list(
    //             &context.clone(),
    //             &vector_length,
    //             &element_list,
    //             &First::<usize, usize, usize>::new(),
    //         )
    //         .unwrap();
    //         let mut product_vector = SparseVector::<usize>::new(&context, &vector_length).unwrap();

    //         let operator = BinaryOperatorApplier::new(
    //             &Plus::<usize, usize, usize>::new(),
    //             &OperatorOptions::new_default(),
    //             None,
    //         );

    //         operator
    //             .apply_with_vector_as_second_argument(&10, &vector, &mut product_vector)
    //             .unwrap();

    //         assert_eq!(product_vector.number_of_stored_elements().unwrap(), 5);
    //         assert_eq!(product_vector.get_element_value(&(10+i)).unwrap(), i+10);
    //     }
    // }
}
