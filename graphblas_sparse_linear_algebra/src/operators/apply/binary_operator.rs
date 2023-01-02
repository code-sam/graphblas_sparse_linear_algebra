use std::marker::PhantomData;
use std::ptr;

use suitesparse_graphblas_sys::{
    GrB_Matrix_apply_BinaryOp1st_BOOL, GrB_Matrix_apply_BinaryOp1st_FP32,
    GrB_Matrix_apply_BinaryOp1st_FP64, GrB_Matrix_apply_BinaryOp1st_INT16,
    GrB_Matrix_apply_BinaryOp1st_INT32, GrB_Matrix_apply_BinaryOp1st_INT64,
    GrB_Matrix_apply_BinaryOp1st_INT8, GrB_Matrix_apply_BinaryOp1st_UINT16,
    GrB_Matrix_apply_BinaryOp1st_UINT32, GrB_Matrix_apply_BinaryOp1st_UINT64,
    GrB_Matrix_apply_BinaryOp1st_UINT8, GrB_Matrix_apply_BinaryOp2nd_BOOL,
    GrB_Matrix_apply_BinaryOp2nd_FP32, GrB_Matrix_apply_BinaryOp2nd_FP64,
    GrB_Matrix_apply_BinaryOp2nd_INT16, GrB_Matrix_apply_BinaryOp2nd_INT32,
    GrB_Matrix_apply_BinaryOp2nd_INT64, GrB_Matrix_apply_BinaryOp2nd_INT8,
    GrB_Matrix_apply_BinaryOp2nd_UINT16, GrB_Matrix_apply_BinaryOp2nd_UINT32,
    GrB_Matrix_apply_BinaryOp2nd_UINT64, GrB_Matrix_apply_BinaryOp2nd_UINT8,
    GrB_Vector_apply_BinaryOp1st_BOOL, GrB_Vector_apply_BinaryOp1st_FP32,
    GrB_Vector_apply_BinaryOp1st_FP64, GrB_Vector_apply_BinaryOp1st_INT16,
    GrB_Vector_apply_BinaryOp1st_INT32, GrB_Vector_apply_BinaryOp1st_INT64,
    GrB_Vector_apply_BinaryOp1st_INT8, GrB_Vector_apply_BinaryOp1st_UINT16,
    GrB_Vector_apply_BinaryOp1st_UINT32, GrB_Vector_apply_BinaryOp1st_UINT64,
    GrB_Vector_apply_BinaryOp1st_UINT8, GrB_Vector_apply_BinaryOp2nd_BOOL,
    GrB_Vector_apply_BinaryOp2nd_FP32, GrB_Vector_apply_BinaryOp2nd_FP64,
    GrB_Vector_apply_BinaryOp2nd_INT16, GrB_Vector_apply_BinaryOp2nd_INT32,
    GrB_Vector_apply_BinaryOp2nd_INT64, GrB_Vector_apply_BinaryOp2nd_INT8,
    GrB_Vector_apply_BinaryOp2nd_UINT16, GrB_Vector_apply_BinaryOp2nd_UINT32,
    GrB_Vector_apply_BinaryOp2nd_UINT64, GrB_Vector_apply_BinaryOp2nd_UINT8,
};

use crate::collections::sparse_matrix::{GraphblasSparseMatrixTrait, SparseMatrix};
use crate::collections::sparse_vector::{GraphblasSparseVectorTrait, SparseVector};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_4_typed_graphblas_functions_with_implementation_type;
use crate::value_type::{AsBoolean, ConvertScalar, ValueType};

use crate::bindings_to_graphblas_implementation::{GrB_BinaryOp, GrB_Descriptor};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mutable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<
        FirstArgument: ValueType,
        SecondArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > Send for BinaryOperatorApplier<FirstArgument, SecondArgument, Product, EvaluationDomain>
{
}
unsafe impl<
        FirstArgument: ValueType,
        SecondArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > Sync for BinaryOperatorApplier<FirstArgument, SecondArgument, Product, EvaluationDomain>
{
}

#[derive(Debug, Clone)]
pub struct BinaryOperatorApplier<
    FirstArgument: ValueType,
    SecondArgument: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
> {
    _first_argument: PhantomData<FirstArgument>,
    _second_argument: PhantomData<SecondArgument>,
    _result: PhantomData<Product>,
    _evaluation_domain: PhantomData<EvaluationDomain>,

    binary_operator: GrB_BinaryOp,
    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<
        FirstArgument: ValueType,
        SecondArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > BinaryOperatorApplier<FirstArgument, SecondArgument, Product, EvaluationDomain>
{
    pub fn new(
        binary_operator: &dyn BinaryOperator<
            FirstArgument,
            SecondArgument,
            Product,
            EvaluationDomain,
        >,
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<Product, Product, Product, EvaluationDomain>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
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
            _evaluation_domain: PhantomData,
        }
    }

    pub(crate) unsafe fn binary_operator(&self) -> GrB_BinaryOp {
        self.binary_operator
    }
    pub(crate) unsafe fn accumulator(&self) -> GrB_BinaryOp {
        self.accumulator
    }
    pub(crate) unsafe fn options(&self) -> GrB_Descriptor {
        self.options
    }
}

pub trait ApplyBinaryOperator<FirstArgument, SecondArgument, Product, EvaluationDomain>
where
    FirstArgument: ValueType,
    SecondArgument: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
{
    fn apply_with_vector_as_first_argument(
        &self,
        first_argument: &SparseVector<FirstArgument>,
        second_argument: &EvaluationDomain,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_second_argument(
        &self,
        first_argument: &EvaluationDomain,
        second_argument: &SparseVector<SecondArgument>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_first_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseVector<FirstArgument>,
        second_argument: &EvaluationDomain,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_vector_as_second_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &EvaluationDomain,
        second_argument: &SparseVector<SecondArgument>,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_first_argument(
        &self,
        first_argument: &SparseMatrix<FirstArgument>,
        second_argument: &EvaluationDomain,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_second_argument(
        &self,
        first_argument: &EvaluationDomain,
        second_argument: &SparseMatrix<SecondArgument>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_first_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &SparseMatrix<FirstArgument>,
        second_argument: &EvaluationDomain,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_matrix_as_second_argument_and_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        first_argument: &EvaluationDomain,
        second_argument: &SparseMatrix<SecondArgument>,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_apply_binary_operator {
    ($value_type: ty, $_implementation_type: ty, $graphblas_function_1: ident, $graphblas_function_2: ident, $graphblas_function_3: ident, $graphblas_function_4: ident) => {
        impl<FirstArgument: ValueType, SecondArgument: ValueType, Product: ValueType>
            ApplyBinaryOperator<FirstArgument, SecondArgument, Product, $value_type>
            for BinaryOperatorApplier<FirstArgument, SecondArgument, Product, $value_type>
        {
            fn apply_with_vector_as_first_argument(
                &self,
                first_argument: &SparseVector<FirstArgument>,
                second_argument: &$value_type,
                product: &mut SparseVector<Product>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let second_argument = second_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_1(
                            product.graphblas_vector(),
                            ptr::null_mut(),
                            self.accumulator,
                            self.binary_operator,
                            first_argument.graphblas_vector(),
                            second_argument,
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_vector_as_second_argument(
                &self,
                first_argument: &$value_type,
                second_argument: &SparseVector<SecondArgument>,
                product: &mut SparseVector<Product>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let first_argument = first_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_2(
                            product.graphblas_vector(),
                            ptr::null_mut(),
                            self.accumulator,
                            self.binary_operator,
                            first_argument,
                            second_argument.graphblas_vector(),
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_vector_as_first_argument_and_mask<
                MaskValueType: ValueType + AsBoolean,
            >(
                &self,
                first_argument: &SparseVector<FirstArgument>,
                second_argument: &$value_type,
                product: &mut SparseVector<Product>,
                mask: &SparseVector<MaskValueType>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let second_argument = second_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_1(
                            product.graphblas_vector(),
                            mask.graphblas_vector(),
                            self.accumulator,
                            self.binary_operator,
                            first_argument.graphblas_vector(),
                            second_argument,
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_vector_as_second_argument_and_mask<
                MaskValueType: ValueType + AsBoolean,
            >(
                &self,
                first_argument: &$value_type,
                second_argument: &SparseVector<SecondArgument>,
                product: &mut SparseVector<Product>,
                mask: &SparseVector<MaskValueType>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let first_argument = first_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_2(
                            product.graphblas_vector(),
                            mask.graphblas_vector(),
                            self.accumulator,
                            self.binary_operator,
                            first_argument,
                            second_argument.graphblas_vector(),
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_first_argument(
                &self,
                first_argument: &SparseMatrix<FirstArgument>,
                second_argument: &$value_type,
                product: &mut SparseMatrix<Product>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let second_argument = second_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_3(
                            product.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            self.binary_operator,
                            first_argument.graphblas_matrix(),
                            second_argument,
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_second_argument(
                &self,
                first_argument: &$value_type,
                second_argument: &SparseMatrix<SecondArgument>,
                product: &mut SparseMatrix<Product>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let first_argument = first_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_4(
                            product.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            self.binary_operator,
                            first_argument,
                            second_argument.graphblas_matrix(),
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_first_argument_and_mask<
                MaskValueType: ValueType + AsBoolean,
            >(
                &self,
                first_argument: &SparseMatrix<FirstArgument>,
                second_argument: &$value_type,
                product: &mut SparseMatrix<Product>,
                mask: &SparseMatrix<MaskValueType>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let second_argument = second_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_3(
                            product.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            self.accumulator,
                            self.binary_operator,
                            first_argument.graphblas_matrix(),
                            second_argument,
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }

            fn apply_with_matrix_as_second_argument_and_mask<
                MaskValueType: ValueType + AsBoolean,
            >(
                &self,
                first_argument: &$value_type,
                second_argument: &SparseMatrix<SecondArgument>,
                product: &mut SparseMatrix<Product>,
                mask: &SparseMatrix<MaskValueType>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let first_argument = first_argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_4(
                            product.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            self.accumulator,
                            self.binary_operator,
                            first_argument,
                            second_argument.graphblas_matrix(),
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_4_typed_graphblas_functions_with_implementation_type!(
    implement_apply_binary_operator,
    GrB_Vector_apply_BinaryOp2nd,
    GrB_Vector_apply_BinaryOp1st,
    GrB_Matrix_apply_BinaryOp2nd,
    GrB_Matrix_apply_BinaryOp1st
);

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

        let second_agrument = 10;
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
        let first_agrument = 10;
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

        let second_agrument = 10;
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
        let first_argument = 10;
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

        let second_agrument = 10;
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
        let first_agrument = 10;
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
            &Plus::<u8, bool, i8, bool>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        let second_argument = true;
        operator
            .apply_with_vector_as_first_argument(&vector, &second_argument, &mut product_vector)
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 1);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        let operator = BinaryOperatorApplier::new(
            &Plus::<u8, bool, i8, bool>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        operator
            .apply_with_vector_as_first_argument(&vector, &second_argument, &mut product_vector)
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
