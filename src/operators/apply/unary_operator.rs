use std::marker::PhantomData;
use std::ptr;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator,
    mask::{MatrixMask, VectorMask},
    options::OperatorOptions,
    unary_operator::UnaryOperator,
};
use crate::sparse_matrix::SparseMatrix;
use crate::sparse_vector::SparseVector;
use crate::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_apply, GrB_UnaryOp, GrB_Vector_apply,
};

#[derive(Debug, Clone)]
pub struct UnaryOperatorApplier<T: ValueType> {
    _result: PhantomData<T>,

    unary_operator: GrB_UnaryOp,
    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<T: ValueType> UnaryOperatorApplier<T> {
    pub fn new(
        unary_operator: &dyn UnaryOperator<T>,
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<T, T, T>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            unary_operator: unary_operator.graphblas_type(),
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _result: PhantomData,
        }
    }
}

pub trait UnaryOperatorApplierTrait<Argument, Product>
where
    Argument: ValueType,
    Product: ValueType,
{
    fn apply_to_vector(
        &self,
        argument: &SparseVector<Argument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_vector_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseVector<Argument>,
        product: &mut SparseVector<Product>,
        mask: &VectorMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix(
        &self,
        argument: &SparseMatrix<Argument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseMatrix<Argument>,
        product: &mut SparseMatrix<Product>,
        mask: &MatrixMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_unary_operator {
    ($argument_type:ty, $product_type:ty) => {
        impl UnaryOperatorApplierTrait<$argument_type, $product_type>
            for UnaryOperatorApplier<$product_type>
        {
            fn apply_to_vector(
                &self,
                argument: &SparseVector<$argument_type>,
                product: &mut SparseVector<$product_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();

                context.call(|| unsafe {
                    GrB_Vector_apply(
                        product.graphblas_vector(),
                        ptr::null_mut(),
                        self.accumulator,
                        self.unary_operator,
                        argument.graphblas_vector(),
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_to_vector_with_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                argument: &SparseVector<$argument_type>,
                product: &mut SparseVector<$product_type>,
                mask: &VectorMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();

                context.call(|| unsafe {
                    GrB_Vector_apply(
                        product.graphblas_vector(),
                        mask.graphblas_vector(),
                        self.accumulator,
                        self.unary_operator,
                        argument.graphblas_vector(),
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_to_matrix(
                &self,
                argument: &SparseMatrix<$argument_type>,
                product: &mut SparseMatrix<$product_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();

                context.call(|| unsafe {
                    GrB_Matrix_apply(
                        product.graphblas_matrix(),
                        ptr::null_mut(),
                        self.accumulator,
                        self.unary_operator,
                        argument.graphblas_matrix(),
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_to_matrix_with_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                argument: &SparseMatrix<$argument_type>,
                product: &mut SparseMatrix<$product_type>,
                mask: &MatrixMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();

                context.call(|| unsafe {
                    GrB_Matrix_apply(
                        product.graphblas_matrix(),
                        mask.graphblas_matrix(),
                        self.accumulator,
                        self.unary_operator,
                        argument.graphblas_matrix(),
                        self.options,
                    )
                })?;

                Ok(())
            }
        }
    };
}

implement_unary_operator!(bool, bool);
implement_unary_operator!(u8, u8);
implement_unary_operator!(u16, u16);
implement_unary_operator!(u32, u32);
implement_unary_operator!(u64, u64);
implement_unary_operator!(i8, i8);
implement_unary_operator!(i16, i16);
implement_unary_operator!(i32, i32);
implement_unary_operator!(i64, i64);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::operators::unary_operator::{Identity, One};
    use crate::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };
    use crate::sparse_vector::{FromVectorElementList, GetVectorElementValue, VectorElementList};

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
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let operator =
            UnaryOperatorApplier::new(&One::<u8>::new(), &OperatorOptions::new_default(), None);

        operator
            .apply_to_matrix(&matrix, &mut product_matrix)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.get_element_value(&(2, 1).into()).unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(9, 1).into()).unwrap(), 0);

        let operator = UnaryOperatorApplier::new(
            &Identity::<u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );
        operator
            .apply_to_matrix(&matrix, &mut product_matrix)
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.get_element_value(&(2, 1).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(9, 1).into()).unwrap(), 0);
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
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator =
            UnaryOperatorApplier::new(&One::<u8>::new(), &OperatorOptions::new_default(), None);

        operator
            .apply_to_vector(&vector, &mut product_vector)
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 1);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        let operator = UnaryOperatorApplier::new(
            &Identity::<u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );
        operator
            .apply_to_vector(&vector, &mut product_vector)
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    }
}
