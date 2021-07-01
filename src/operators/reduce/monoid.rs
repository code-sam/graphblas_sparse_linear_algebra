use std::ptr;

use std::marker::PhantomData;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{BinaryOperator, Monoid, OperatorOptions, VectorMask};

use crate::sparse_matrix::SparseMatrix;
use crate::sparse_vector::SparseVector;

use crate::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_reduce_BOOL, GrB_Matrix_reduce_FP32,
    GrB_Matrix_reduce_FP64, GrB_Matrix_reduce_INT16, GrB_Matrix_reduce_INT32,
    GrB_Matrix_reduce_INT64, GrB_Matrix_reduce_INT8, GrB_Matrix_reduce_Monoid,
    GrB_Matrix_reduce_UINT16, GrB_Matrix_reduce_UINT32, GrB_Matrix_reduce_UINT64,
    GrB_Matrix_reduce_UINT8, GrB_Monoid, GrB_Vector_reduce_BOOL, GrB_Vector_reduce_FP32,
    GrB_Vector_reduce_FP64, GrB_Vector_reduce_INT16, GrB_Vector_reduce_INT32,
    GrB_Vector_reduce_INT64, GrB_Vector_reduce_INT8, GrB_Vector_reduce_UINT16,
    GrB_Vector_reduce_UINT32, GrB_Vector_reduce_UINT64, GrB_Vector_reduce_UINT8,
};

#[derive(Debug, Clone)]
pub struct MonoidReducer<T: ValueType> {
    _value: PhantomData<T>,

    monoid: GrB_Monoid,
    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

pub trait MonoidScalarReducer<T>
where
    T: ValueType,
{
    fn matrix_to_scalar(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn vector_to_scalar(
        &self,
        argument: &SparseVector<T>,
        product: &mut T,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> MonoidReducer<T> {
    pub fn new(
        monoid: &dyn Monoid<T>,
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<T, T, T>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            monoid: monoid.graphblas_type(),
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _value: PhantomData,
        }
    }

    pub fn to_vector(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseVector<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_reduce_Monoid(
                product.graphblas_vector(),
                ptr::null_mut(),
                self.accumulator,
                self.monoid,
                argument.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }

    pub fn to_vector_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseVector<T>,
        mask: &VectorMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_reduce_Monoid(
                product.graphblas_vector(),
                mask.graphblas_vector(),
                self.accumulator,
                self.monoid,
                argument.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }
}

macro_rules! implement_monoid_reducer {
    ($value_type:ty, $matrix_reducer_operator:ident, $vector_reducer_operator:ident) => {
        impl MonoidScalarReducer<$value_type> for MonoidReducer<$value_type> {
            fn matrix_to_scalar(
                &self,
                argument: &SparseMatrix<$value_type>,
                product: &mut $value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();

                context.call(|| unsafe {
                    $matrix_reducer_operator(
                        product,
                        self.accumulator,
                        self.monoid,
                        argument.graphblas_matrix(),
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn vector_to_scalar(
                &self,
                argument: &SparseVector<$value_type>,
                product: &mut $value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();

                context.call(|| unsafe {
                    $vector_reducer_operator(
                        product,
                        self.accumulator,
                        self.monoid,
                        argument.graphblas_vector(),
                        self.options,
                    )
                })?;

                Ok(())
            }
        }
    };
}

implement_monoid_reducer!(bool, GrB_Matrix_reduce_BOOL, GrB_Vector_reduce_BOOL);
implement_monoid_reducer!(u8, GrB_Matrix_reduce_UINT8, GrB_Vector_reduce_UINT8);
implement_monoid_reducer!(u16, GrB_Matrix_reduce_UINT16, GrB_Vector_reduce_UINT16);
implement_monoid_reducer!(u32, GrB_Matrix_reduce_UINT32, GrB_Vector_reduce_UINT32);
implement_monoid_reducer!(u64, GrB_Matrix_reduce_UINT64, GrB_Vector_reduce_UINT64);
implement_monoid_reducer!(i8, GrB_Matrix_reduce_INT8, GrB_Vector_reduce_INT8);
implement_monoid_reducer!(i16, GrB_Matrix_reduce_INT16, GrB_Vector_reduce_INT16);
implement_monoid_reducer!(i32, GrB_Matrix_reduce_INT32, GrB_Vector_reduce_INT32);
implement_monoid_reducer!(i64, GrB_Matrix_reduce_INT64, GrB_Vector_reduce_INT64);
implement_monoid_reducer!(f32, GrB_Matrix_reduce_FP32, GrB_Vector_reduce_FP32);
implement_monoid_reducer!(f64, GrB_Matrix_reduce_FP64, GrB_Vector_reduce_FP64);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::operators::monoid::Plus as MonoidPlus;

    use crate::sparse_matrix::{FromMatrixElementList, MatrixElementList, Size};
    use crate::sparse_vector::{FromVectorElementList, GetVectorElementValue, VectorElementList};

    #[test]
    fn test_monoid_to_vector_reducer() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (1, 5, 1).into(),
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

        let mut product_vector =
            SparseVector::<u8>::new(&context, &matrix_size.row_height()).unwrap();

        let reducer = MonoidReducer::new(
            &MonoidPlus::<u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        reducer.to_vector(&matrix, &mut product_vector).unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        let mask_element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            // (5, 5).into(),
        ]);

        let mask = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &matrix_size.row_height(),
            &mask_element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_vector =
            SparseVector::<u8>::new(&context, &matrix_size.row_height()).unwrap();

        reducer
            .to_vector_with_mask(&matrix, &mut product_vector, &mask.into())
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_vector.get_element_value(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&5).unwrap(), 0);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    }

    #[test]
    fn test_monoid_to_scalar_reducer_for_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (1, 5, 1).into(),
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

        let mut product = 0;

        let reducer = MonoidReducer::new(
            &MonoidPlus::<u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        reducer.matrix_to_scalar(&matrix, &mut product).unwrap();

        println!("{}", product);

        assert_eq!(product, 13);
    }

    #[test]
    fn test_monoid_to_scalar_reducer_for_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector_length = 10;
        let vector = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product = 0;

        let reducer = MonoidReducer::new(
            &MonoidPlus::<u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        reducer.vector_to_scalar(&vector, &mut product).unwrap();

        println!("{}", product);

        assert_eq!(product, 12);
    }
}
