use std::marker::PhantomData;
use std::ptr;

use suitesparse_graphblas_sys::{
    GrB_IndexUnaryOp, GrB_Matrix_apply_IndexOp_BOOL, GrB_Matrix_apply_IndexOp_FP32,
    GrB_Matrix_apply_IndexOp_FP64, GrB_Matrix_apply_IndexOp_INT16, GrB_Matrix_apply_IndexOp_INT32,
    GrB_Matrix_apply_IndexOp_INT64, GrB_Matrix_apply_IndexOp_INT8, GrB_Matrix_apply_IndexOp_UINT16,
    GrB_Matrix_apply_IndexOp_UINT32, GrB_Matrix_apply_IndexOp_UINT64,
    GrB_Matrix_apply_IndexOp_UINT8, GrB_Vector_apply_IndexOp_BOOL, GrB_Vector_apply_IndexOp_FP32,
    GrB_Vector_apply_IndexOp_FP64, GrB_Vector_apply_IndexOp_INT16, GrB_Vector_apply_IndexOp_INT32,
    GrB_Vector_apply_IndexOp_INT64, GrB_Vector_apply_IndexOp_INT8, GrB_Vector_apply_IndexOp_UINT16,
    GrB_Vector_apply_IndexOp_UINT32, GrB_Vector_apply_IndexOp_UINT64,
    GrB_Vector_apply_IndexOp_UINT8,
};

use crate::collections::sparse_matrix::{GraphblasSparseMatrixTrait, SparseMatrix};
use crate::collections::sparse_vector::{GraphblasSparseVectorTrait, SparseVector};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_2_typed_graphblas_functions_with_implementation_type;
use crate::value_type::{AsBoolean, ConvertScalar, ValueType};

use crate::bindings_to_graphblas_implementation::{GrB_BinaryOp, GrB_Descriptor};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<
        ApplyTo: ValueType,
        OperatorArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > Send for IndexUnaryOperatorApplier<ApplyTo, OperatorArgument, Product, EvaluationDomain>
{
}
unsafe impl<
        ApplyTo: ValueType,
        OperatorArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > Sync for IndexUnaryOperatorApplier<ApplyTo, OperatorArgument, Product, EvaluationDomain>
{
}

#[derive(Debug, Clone)]
pub struct IndexUnaryOperatorApplier<
    ApplyTo: ValueType,
    OperatorArgument: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
> {
    _first_argument: PhantomData<ApplyTo>,
    _second_argument: PhantomData<OperatorArgument>,
    _product: PhantomData<Product>,
    _evaluation_domain: PhantomData<EvaluationDomain>,

    index_unary_operator: GrB_IndexUnaryOp,
    accumulator: GrB_BinaryOp,
    options: GrB_Descriptor,
}

impl<
        ApplyTo: ValueType,
        OperatorArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > IndexUnaryOperatorApplier<ApplyTo, OperatorArgument, Product, EvaluationDomain>
{
    pub fn new(
        index_unary_operator: &impl IndexUnaryOperator<
            ApplyTo,
            OperatorArgument,
            Product,
            EvaluationDomain,
        >,
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<Product, Product, Product, Product>,
    ) -> Self {
        Self {
            index_unary_operator: index_unary_operator.graphblas_type(),
            accumulator: accumulator.accumulator_graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _first_argument: PhantomData,
            _second_argument: PhantomData,
            _product: PhantomData,
            _evaluation_domain: PhantomData,
        }
    }

    pub(crate) unsafe fn index_unary_operator(&self) -> GrB_IndexUnaryOp {
        self.index_unary_operator
    }
    pub(crate) unsafe fn accumulator(&self) -> GrB_BinaryOp {
        self.accumulator
    }
    pub(crate) unsafe fn options(&self) -> GrB_Descriptor {
        self.options
    }
}

pub trait ApplyIndexUnaryOperator<ApplyTo, OperatorArgument, Product, EvaluationDomain>
where
    ApplyTo: ValueType,
    OperatorArgument: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
{
    fn apply_to_vector(
        &self,
        vector: &SparseVector<ApplyTo>,
        argument: &EvaluationDomain,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_vector_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        vector: &SparseVector<ApplyTo>,
        argument: &EvaluationDomain,
        product: &mut SparseVector<Product>,
        mask: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix(
        &self,
        matrix: &SparseMatrix<ApplyTo>,
        argument: &EvaluationDomain,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_to_matrix_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        matrix: &SparseMatrix<ApplyTo>,
        argument: &EvaluationDomain,
        product: &mut SparseMatrix<Product>,
        mask: &SparseMatrix<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_apply_index_binary_operator {
    ($value_type: ty, $_implementation_type: ty, $graphblas_function_1: ident, $graphblas_function_2: ident) => {
        impl<ApplyTo: ValueType, OperatorArgument: ValueType, Product: ValueType>
            ApplyIndexUnaryOperator<ApplyTo, OperatorArgument, Product, $value_type>
            for IndexUnaryOperatorApplier<ApplyTo, OperatorArgument, Product, $value_type>
        {
            fn apply_to_vector(
                &self,
                vector: &SparseVector<ApplyTo>,
                argument: &$value_type,
                product: &mut SparseVector<Product>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let argument = argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_1(
                            product.graphblas_vector(),
                            ptr::null_mut(),
                            self.accumulator,
                            self.index_unary_operator,
                            vector.graphblas_vector(),
                            argument,
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_to_vector_with_mask<MaskValueType: ValueType + AsBoolean>(
                &self,
                vector: &SparseVector<ApplyTo>,
                argument: &$value_type,
                product: &mut SparseVector<Product>,
                mask: &SparseVector<MaskValueType>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let argument = argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_1(
                            product.graphblas_vector(),
                            mask.graphblas_vector(),
                            self.accumulator,
                            self.index_unary_operator,
                            vector.graphblas_vector(),
                            argument,
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_vector() },
                )?;

                Ok(())
            }

            fn apply_to_matrix(
                &self,
                matrix: &SparseMatrix<ApplyTo>,
                argument: &$value_type,
                product: &mut SparseMatrix<Product>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let argument = argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_2(
                            product.graphblas_matrix(),
                            ptr::null_mut(),
                            self.accumulator,
                            self.index_unary_operator,
                            matrix.graphblas_matrix(),
                            argument,
                            self.options,
                        )
                    },
                    unsafe { &product.graphblas_matrix() },
                )?;

                Ok(())
            }

            fn apply_to_matrix_with_mask<MaskValueType: ValueType + AsBoolean>(
                &self,
                matrix: &SparseMatrix<ApplyTo>,
                argument: &$value_type,
                product: &mut SparseMatrix<Product>,
                mask: &SparseMatrix<MaskValueType>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let argument = argument.clone().to_type()?;

                context.call(
                    || unsafe {
                        $graphblas_function_2(
                            product.graphblas_matrix(),
                            mask.graphblas_matrix(),
                            self.accumulator,
                            self.index_unary_operator,
                            matrix.graphblas_matrix(),
                            argument,
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

implement_1_type_macro_for_all_value_types_and_2_typed_graphblas_functions_with_implementation_type!(
    implement_apply_index_binary_operator,
    GrB_Vector_apply_IndexOp,
    GrB_Matrix_apply_IndexOp
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};
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

        let argument = 2i8;

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
            &IsValueGreaterThan::<u8, i8, f32, i8>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        operator
            .apply_to_matrix(&matrix, &argument, &mut product_matrix)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix.get_element_value(&(1, 1).into()).unwrap(),
            Some(0.0)
        );
        assert_eq!(
            product_matrix.get_element_value(&(2, 1).into()).unwrap(),
            Some(0.0)
        );
        assert_eq!(
            product_matrix
                .get_element_value(&(4, 2).into())
                .unwrap()
                .unwrap_or_default(),
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
