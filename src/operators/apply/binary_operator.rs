use std::marker::PhantomData;
use std::ptr;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator,
    mask::{MatrixMask, VectorMask},
    options::OperatorOptions,
};
use crate::value_types::sparse_matrix::SparseMatrix;
use crate::value_types::sparse_vector::SparseVector;
use crate::value_types::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_apply_BinaryOp1st_BOOL,
    GrB_Matrix_apply_BinaryOp1st_FP32, GrB_Matrix_apply_BinaryOp1st_FP64,
    GrB_Matrix_apply_BinaryOp1st_INT16, GrB_Matrix_apply_BinaryOp1st_INT32,
    GrB_Matrix_apply_BinaryOp1st_INT64, GrB_Matrix_apply_BinaryOp1st_INT8,
    GrB_Matrix_apply_BinaryOp1st_UINT16, GrB_Matrix_apply_BinaryOp1st_UINT32,
    GrB_Matrix_apply_BinaryOp1st_UINT64, GrB_Matrix_apply_BinaryOp1st_UINT8,
    GrB_Matrix_apply_BinaryOp2nd_BOOL, GrB_Matrix_apply_BinaryOp2nd_FP32,
    GrB_Matrix_apply_BinaryOp2nd_FP64, GrB_Matrix_apply_BinaryOp2nd_INT16,
    GrB_Matrix_apply_BinaryOp2nd_INT32, GrB_Matrix_apply_BinaryOp2nd_INT64,
    GrB_Matrix_apply_BinaryOp2nd_INT8, GrB_Matrix_apply_BinaryOp2nd_UINT16,
    GrB_Matrix_apply_BinaryOp2nd_UINT32, GrB_Matrix_apply_BinaryOp2nd_UINT64,
    GrB_Matrix_apply_BinaryOp2nd_UINT8, GrB_Vector_apply_BinaryOp1st_BOOL,
    GrB_Vector_apply_BinaryOp1st_FP32, GrB_Vector_apply_BinaryOp1st_FP64,
    GrB_Vector_apply_BinaryOp1st_INT16, GrB_Vector_apply_BinaryOp1st_INT32,
    GrB_Vector_apply_BinaryOp1st_INT64, GrB_Vector_apply_BinaryOp1st_INT8,
    GrB_Vector_apply_BinaryOp1st_UINT16, GrB_Vector_apply_BinaryOp1st_UINT32,
    GrB_Vector_apply_BinaryOp1st_UINT64, GrB_Vector_apply_BinaryOp1st_UINT8,
    GrB_Vector_apply_BinaryOp2nd_BOOL, GrB_Vector_apply_BinaryOp2nd_FP32,
    GrB_Vector_apply_BinaryOp2nd_FP64, GrB_Vector_apply_BinaryOp2nd_INT16,
    GrB_Vector_apply_BinaryOp2nd_INT32, GrB_Vector_apply_BinaryOp2nd_INT64,
    GrB_Vector_apply_BinaryOp2nd_INT8, GrB_Vector_apply_BinaryOp2nd_UINT16,
    GrB_Vector_apply_BinaryOp2nd_UINT32, GrB_Vector_apply_BinaryOp2nd_UINT64,
    GrB_Vector_apply_BinaryOp2nd_UINT8,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for BinaryOperatorApplier<bool, bool, bool> {}
unsafe impl Send for BinaryOperatorApplier<u8, u8, u8> {}
unsafe impl Send for BinaryOperatorApplier<u16, u16, u16> {}
unsafe impl Send for BinaryOperatorApplier<u32, u32, u32> {}
unsafe impl Send for BinaryOperatorApplier<u64, u64, u64> {}
unsafe impl Send for BinaryOperatorApplier<i8, i8, i8> {}
unsafe impl Send for BinaryOperatorApplier<i16, i16, i16> {}
unsafe impl Send for BinaryOperatorApplier<i32, i32, i32> {}
unsafe impl Send for BinaryOperatorApplier<i64, i64, i64> {}
unsafe impl Send for BinaryOperatorApplier<f32, f32, f32> {}
unsafe impl Send for BinaryOperatorApplier<f64, f64, f64> {}

unsafe impl Sync for BinaryOperatorApplier<bool, bool, bool> {}
unsafe impl Sync for BinaryOperatorApplier<u8, u8, u8> {}
unsafe impl Sync for BinaryOperatorApplier<u16, u16, u16> {}
unsafe impl Sync for BinaryOperatorApplier<u32, u32, u32> {}
unsafe impl Sync for BinaryOperatorApplier<u64, u64, u64> {}
unsafe impl Sync for BinaryOperatorApplier<i8, i8, i8> {}
unsafe impl Sync for BinaryOperatorApplier<i16, i16, i16> {}
unsafe impl Sync for BinaryOperatorApplier<i32, i32, i32> {}
unsafe impl Sync for BinaryOperatorApplier<i64, i64, i64> {}
unsafe impl Sync for BinaryOperatorApplier<f32, f32, f32> {}
unsafe impl Sync for BinaryOperatorApplier<f64, f64, f64> {}

#[derive(Debug, Clone)]
pub struct BinaryOperatorApplier<
    FirstArgument: ValueType,
    SecondArgument: ValueType,
    Product: ValueType,
> {
    _first_argument: PhantomData<FirstArgument>,
    _second_argument: PhantomData<SecondArgument>,
    _result: PhantomData<Product>,

    binary_operator: GrB_BinaryOp,
    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<FirstArgument: ValueType, SecondArgument: ValueType, Product: ValueType>
    BinaryOperatorApplier<FirstArgument, SecondArgument, Product>
{
    pub fn new(
        binary_operator: &dyn BinaryOperator<FirstArgument, SecondArgument, Product>,
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<FirstArgument, SecondArgument, Product>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            binary_operator: binary_operator.graphblas_type(),
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _first_argument: PhantomData,
            _second_argument: PhantomData,
            _result: PhantomData,
        }
    }
}

pub trait BinaryOperatorApplierTrait<FirstArgument, SecondArgument, Product>
where
    FirstArgument: ValueType,
    SecondArgument: ValueType,
    Product: ValueType,
{
    fn apply_with_vector_as_first_argument(
        &self,
        first_argument: &SparseVector<FirstArgument>,
        second_argument: &SecondArgument,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_second_argument(
        &self,
        first_argument: &FirstArgument,
        second_argument: &SparseVector<SecondArgument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_first_argument_and_mask<
        MaskValueType: ValueType,
        AsBool: AsBoolean<MaskValueType>,
    >(
        &self,
        first_argument: &SparseVector<FirstArgument>,
        second_argument: &SecondArgument,
        product: &mut SparseVector<Product>,
        mask: &VectorMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_second_argument_and_mask<
        MaskValueType: ValueType,
        AsBool: AsBoolean<MaskValueType>,
    >(
        &self,
        first_argument: &FirstArgument,
        second_argument: &SparseVector<SecondArgument>,
        product: &mut SparseVector<Product>,
        mask: &VectorMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_first_argument(
        &self,
        first_argument: &SparseMatrix<FirstArgument>,
        second_argument: &SecondArgument,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_second_argument(
        &self,
        first_argument: &FirstArgument,
        second_argument: &SparseMatrix<SecondArgument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_first_argument_and_mask<
        MaskValueType: ValueType,
        AsBool: AsBoolean<MaskValueType>,
    >(
        &self,
        first_argument: &SparseMatrix<FirstArgument>,
        second_argument: &SecondArgument,
        product: &mut SparseMatrix<Product>,
        mask: &MatrixMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_second_argument_and_mask<
        MaskValueType: ValueType,
        AsBool: AsBoolean<MaskValueType>,
    >(
        &self,
        first_argument: &FirstArgument,
        second_argument: &SparseMatrix<SecondArgument>,
        product: &mut SparseMatrix<Product>,
        mask: &MatrixMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_binary_operator {
    ($first_argument_type:ty, $second_argument_type:ty, $product_type:ty, $operator_vector_as_first_argument:ident, $operator_vector_as_second_argument:ident, $operator_matrix_as_first_argument:ident, $operator_matrix_as_second_argument:ident) => {
        impl BinaryOperatorApplierTrait<$first_argument_type, $second_argument_type, $product_type>
            for BinaryOperatorApplier<$first_argument_type, $second_argument_type, $product_type>
        {
            fn apply_with_vector_as_first_argument(
                &self,
                first_argument: &SparseVector<$first_argument_type>,
                second_argument: &$second_argument_type,
                product: &mut SparseVector<$product_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                context.call(|| unsafe {
                    $operator_vector_as_first_argument(
                        product.graphblas_vector(),
                        ptr::null_mut(),
                        self.accumulator,
                        self.binary_operator,
                        first_argument.graphblas_vector(),
                        *second_argument,
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_with_vector_as_second_argument(
                &self,
                first_argument: &$first_argument_type,
                second_argument: &SparseVector<$second_argument_type>,
                product: &mut SparseVector<$product_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                context.call(|| unsafe {
                    $operator_vector_as_second_argument(
                        product.graphblas_vector(),
                        ptr::null_mut(),
                        self.accumulator,
                        self.binary_operator,
                        *first_argument,
                        second_argument.graphblas_vector(),
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_with_vector_as_first_argument_and_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                first_argument: &SparseVector<$first_argument_type>,
                second_argument: &$second_argument_type,
                product: &mut SparseVector<$product_type>,
                mask: &VectorMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                context.call(|| unsafe {
                    $operator_vector_as_first_argument(
                        product.graphblas_vector(),
                        mask.graphblas_vector(),
                        self.accumulator,
                        self.binary_operator,
                        first_argument.graphblas_vector(),
                        *second_argument,
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_with_vector_as_second_argument_and_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                first_argument: &$first_argument_type,
                second_argument: &SparseVector<$second_argument_type>,
                product: &mut SparseVector<$product_type>,
                mask: &VectorMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                context.call(|| unsafe {
                    $operator_vector_as_second_argument(
                        product.graphblas_vector(),
                        mask.graphblas_vector(),
                        self.accumulator,
                        self.binary_operator,
                        *first_argument,
                        second_argument.graphblas_vector(),
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_with_matrix_as_first_argument(
                &self,
                first_argument: &SparseMatrix<$first_argument_type>,
                second_argument: &$second_argument_type,
                product: &mut SparseMatrix<$product_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                let product_with_write_lock = product.get_write_lock()?;
                let first_argument_with_read_lock = first_argument.get_read_lock()?;

                context.call(|| unsafe {
                    $operator_matrix_as_first_argument(
                        *product_with_write_lock,
                        ptr::null_mut(),
                        self.accumulator,
                        self.binary_operator,
                        *first_argument_with_read_lock,
                        *second_argument,
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_with_matrix_as_second_argument(
                &self,
                first_argument: &$first_argument_type,
                second_argument: &SparseMatrix<$second_argument_type>,
                product: &mut SparseMatrix<$product_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                let product_with_write_lock = product.get_write_lock()?;
                let second_argument_with_read_lock = second_argument.get_read_lock()?;

                context.call(|| unsafe {
                    $operator_matrix_as_second_argument(
                        *product_with_write_lock,
                        ptr::null_mut(),
                        self.accumulator,
                        self.binary_operator,
                        *first_argument,
                        *second_argument_with_read_lock,
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_with_matrix_as_first_argument_and_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                first_argument: &SparseMatrix<$first_argument_type>,
                second_argument: &$second_argument_type,
                product: &mut SparseMatrix<$product_type>,
                mask: &MatrixMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                let product_with_write_lock = product.get_write_lock()?;
                let mask_with_read_lock = mask.get_read_lock()?;
                let first_argument_with_read_lock = first_argument.get_read_lock()?;

                context.call(|| unsafe {
                    $operator_matrix_as_first_argument(
                        *product_with_write_lock,
                        *mask_with_read_lock,
                        self.accumulator,
                        self.binary_operator,
                        *first_argument_with_read_lock,
                        *second_argument,
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn apply_with_matrix_as_second_argument_and_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                first_argument: &$first_argument_type,
                second_argument: &SparseMatrix<$second_argument_type>,
                product: &mut SparseMatrix<$product_type>,
                mask: &MatrixMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                let product_with_write_lock = product.get_write_lock()?;
                let mask_with_read_lock = mask.get_read_lock()?;
                let second_argument_with_read_lock = second_argument.get_read_lock()?;

                context.call(|| unsafe {
                    $operator_matrix_as_second_argument(
                        *product_with_write_lock,
                        *mask_with_read_lock,
                        self.accumulator,
                        self.binary_operator,
                        *first_argument,
                        *second_argument_with_read_lock,
                        self.options,
                    )
                })?;

                Ok(())
            }
        }
    };
}

implement_binary_operator!(
    bool,
    bool,
    bool,
    GrB_Vector_apply_BinaryOp2nd_BOOL,
    GrB_Vector_apply_BinaryOp1st_BOOL,
    GrB_Matrix_apply_BinaryOp2nd_BOOL,
    GrB_Matrix_apply_BinaryOp1st_BOOL
);
implement_binary_operator!(
    u8,
    u8,
    u8,
    GrB_Vector_apply_BinaryOp2nd_UINT8,
    GrB_Vector_apply_BinaryOp1st_UINT8,
    GrB_Matrix_apply_BinaryOp2nd_UINT8,
    GrB_Matrix_apply_BinaryOp1st_UINT8
);
implement_binary_operator!(
    u16,
    u16,
    u16,
    GrB_Vector_apply_BinaryOp2nd_UINT16,
    GrB_Vector_apply_BinaryOp1st_UINT16,
    GrB_Matrix_apply_BinaryOp2nd_UINT16,
    GrB_Matrix_apply_BinaryOp1st_UINT16
);
implement_binary_operator!(
    u32,
    u32,
    u32,
    GrB_Vector_apply_BinaryOp2nd_UINT32,
    GrB_Vector_apply_BinaryOp1st_UINT32,
    GrB_Matrix_apply_BinaryOp2nd_UINT32,
    GrB_Matrix_apply_BinaryOp1st_UINT32
);
implement_binary_operator!(
    u64,
    u64,
    u64,
    GrB_Vector_apply_BinaryOp2nd_UINT64,
    GrB_Vector_apply_BinaryOp1st_UINT64,
    GrB_Matrix_apply_BinaryOp2nd_UINT64,
    GrB_Matrix_apply_BinaryOp1st_UINT64
);
implement_binary_operator!(
    i8,
    i8,
    i8,
    GrB_Vector_apply_BinaryOp2nd_INT8,
    GrB_Vector_apply_BinaryOp1st_INT8,
    GrB_Matrix_apply_BinaryOp2nd_INT8,
    GrB_Matrix_apply_BinaryOp1st_INT8
);
implement_binary_operator!(
    i16,
    i16,
    i16,
    GrB_Vector_apply_BinaryOp2nd_INT16,
    GrB_Vector_apply_BinaryOp1st_INT16,
    GrB_Matrix_apply_BinaryOp2nd_INT16,
    GrB_Matrix_apply_BinaryOp1st_INT16
);
implement_binary_operator!(
    i32,
    i32,
    i32,
    GrB_Vector_apply_BinaryOp2nd_INT32,
    GrB_Vector_apply_BinaryOp1st_INT32,
    GrB_Matrix_apply_BinaryOp2nd_INT32,
    GrB_Matrix_apply_BinaryOp1st_INT32
);
implement_binary_operator!(
    i64,
    i64,
    i64,
    GrB_Vector_apply_BinaryOp2nd_INT64,
    GrB_Vector_apply_BinaryOp1st_INT64,
    GrB_Matrix_apply_BinaryOp2nd_INT64,
    GrB_Matrix_apply_BinaryOp1st_INT64
);
implement_binary_operator!(
    f32,
    f32,
    f32,
    GrB_Vector_apply_BinaryOp2nd_FP32,
    GrB_Vector_apply_BinaryOp1st_FP32,
    GrB_Matrix_apply_BinaryOp2nd_FP32,
    GrB_Matrix_apply_BinaryOp1st_FP32
);
implement_binary_operator!(
    f64,
    f64,
    f64,
    GrB_Vector_apply_BinaryOp2nd_FP64,
    GrB_Vector_apply_BinaryOp1st_FP64,
    GrB_Matrix_apply_BinaryOp2nd_FP64,
    GrB_Matrix_apply_BinaryOp1st_FP64
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::value_types::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };
    use crate::value_types::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, VectorElementList,
    };

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
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let operator = BinaryOperatorApplier::new(
            &First::<u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        operator
            .apply_with_matrix_as_first_argument(&matrix, &10, &mut product_matrix)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.get_element_value(&(2, 1).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(9, 1).into()).unwrap(), 0);

        let operator = BinaryOperatorApplier::new(
            &First::<u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );
        operator
            .apply_with_matrix_as_second_argument(&10, &matrix, &mut product_matrix)
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
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new(
            &First::<u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        operator
            .apply_with_vector_as_first_argument(&vector, &10, &mut product_vector)
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        let operator = BinaryOperatorApplier::new(
            &First::<u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );
        operator
            .apply_with_vector_as_second_argument(&10, &vector, &mut product_vector)
            .unwrap();

        println!("{}", vector);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 10);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    }
}
