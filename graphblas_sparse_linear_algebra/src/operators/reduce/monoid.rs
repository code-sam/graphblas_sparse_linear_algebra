use std::convert::TryInto;

use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::graphblas_bindings::{
    GrB_Matrix_reduce_BOOL, GrB_Matrix_reduce_FP32, GrB_Matrix_reduce_FP64,
    GrB_Matrix_reduce_INT16, GrB_Matrix_reduce_INT32, GrB_Matrix_reduce_INT64,
    GrB_Matrix_reduce_INT8, GrB_Matrix_reduce_Monoid, GrB_Matrix_reduce_UINT16,
    GrB_Matrix_reduce_UINT32, GrB_Matrix_reduce_UINT64, GrB_Matrix_reduce_UINT8,
    GrB_Vector_reduce_BOOL, GrB_Vector_reduce_FP32, GrB_Vector_reduce_FP64,
    GrB_Vector_reduce_INT16, GrB_Vector_reduce_INT32, GrB_Vector_reduce_INT64,
    GrB_Vector_reduce_INT8, GrB_Vector_reduce_UINT16, GrB_Vector_reduce_UINT32,
    GrB_Vector_reduce_UINT64, GrB_Vector_reduce_UINT8,
};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::GetGraphblasDescriptor;
use crate::operators::options::MutateOperatorOptions;
use crate::operators::{monoid::Monoid};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    convert_mut_scalar_to_type, identity_conversion,
    implement_macro_for_all_value_types_and_2_typed_graphblas_functions_with_mutable_scalar_type_conversion,
};
use crate::value_type::{ConvertScalar, ValueType};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for MonoidReducer {}
unsafe impl Sync for MonoidReducer {}

#[derive(Debug, Clone)]
pub struct MonoidReducer {}

impl MonoidReducer {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait MonoidVectorReducer<EvaluationDomain: ValueType> {
    fn to_column_vector(
        &self,
        operator: &impl Monoid<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn to_row_vector(
        &self,
        operator: &impl Monoid<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &(impl GetGraphblasDescriptor + MutateOperatorOptions),
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<EvaluationDomain: ValueType> MonoidVectorReducer<EvaluationDomain> for MonoidReducer {
    fn to_column_vector(
        &self,
        operator: &impl Monoid<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_reduce_Monoid(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    accumulator.accumulator_graphblas_type(),
                    operator.graphblas_type(),
                    argument.graphblas_matrix(),
                    options.graphblas_descriptor(),
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
    }

    fn to_row_vector(
        &self,
        operator: &impl Monoid<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &(impl GetGraphblasDescriptor + MutateOperatorOptions),
    ) -> Result<(), SparseLinearAlgebraError> {
        self.to_column_vector(
            operator,
            argument,
            accumulator,
            product,
            mask,
            &options.with_negated_transpose_input0(),
        )
    }
}

pub trait MonoidScalarReducer<EvaluationDomain: ValueType> {
    fn matrix_to_scalar(
        &self,
        operator: &impl Monoid<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseMatrix + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut EvaluationDomain,
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn vector_to_scalar(
        &self,
        operator: &impl Monoid<EvaluationDomain>,
        argument: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut EvaluationDomain,
        options: &impl GetGraphblasDescriptor,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_monoid_reducer {
    ($value_type:ty, $graphblas_implementation_type:ty, $matrix_reducer_operator:ident, $vector_reducer_operator:ident, $convert_to_type:ident) => {
        impl MonoidScalarReducer<$value_type> for MonoidReducer {
            fn matrix_to_scalar(
                &self,
                operator: &impl Monoid<$value_type>,
                argument: &(impl GetGraphblasSparseMatrix + GetContext),
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut $value_type,
                options: &impl GetGraphblasDescriptor,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();
                let mut tmp_product = product.to_owned().to_type()?;

                // TODO: support detailed error information
                context.call_without_detailed_error_information(|| unsafe {
                    $matrix_reducer_operator(
                        &mut tmp_product,
                        accumulator.accumulator_graphblas_type(),
                        operator.graphblas_type(),
                        argument.graphblas_matrix(),
                        options.graphblas_descriptor(),
                    )
                })?;

                $convert_to_type!(tmp_product, $value_type);
                *product = tmp_product;
                Ok(())
            }

            // TODO: support detailed error information
            fn vector_to_scalar(
                &self,
                operator: &impl Monoid<$value_type>,
                argument: &(impl GetGraphblasSparseVector + GetContext),
                accumulator: &impl AccumulatorBinaryOperator<$value_type>,
                product: &mut $value_type,
                options: &impl GetGraphblasDescriptor,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();
                let mut tmp_product = product.to_owned().to_type()?;

                context.call_without_detailed_error_information(|| unsafe {
                    $vector_reducer_operator(
                        &mut tmp_product,
                        accumulator.accumulator_graphblas_type(),
                        operator.graphblas_type(),
                        argument.graphblas_vector(),
                        options.graphblas_descriptor(),
                    )
                })?;

                $convert_to_type!(tmp_product, $value_type);
                *product = tmp_product;
                Ok(())
            }
        }
    };
}

implement_macro_for_all_value_types_and_2_typed_graphblas_functions_with_mutable_scalar_type_conversion!(
    implement_monoid_reducer,
    GrB_Matrix_reduce,
    GrB_Vector_reduce
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::Assignment;
    use crate::operators::binary_operator::First;
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::monoid::Plus as MonoidPlus;

    use crate::collections::sparse_matrix::operations::FromMatrixElementList;
    use crate::collections::sparse_matrix::GetMatrixDimensions;
    use crate::collections::sparse_matrix::{MatrixElementList, Size, SparseMatrix};
    use crate::collections::sparse_vector::operations::FromVectorElementList;
    use crate::collections::sparse_vector::operations::GetVectorElementValue;
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types_except_bool;

    macro_rules! test_monoid {
        ($value_type:ty) => {
            paste::paste! {
                #[test]
                fn [<test_monoid_to_vector_reducer_ $value_type>]() {
                    let context = Context::init_ready(Mode::NonBlocking).unwrap();

                    let element_list = MatrixElementList::<$value_type>::from_element_vector(vec![
                        (1, 1, 1 as $value_type).into(),
                        (1, 5, 1 as $value_type).into(),
                        (2, 1, 2 as $value_type).into(),
                        (4, 2, 4 as $value_type).into(),
                        (5, 2, 5 as $value_type).into(),
                    ]);

                    let matrix_size: Size = (10, 15).into();
                    let matrix = SparseMatrix::<$value_type>::from_element_list(
                        &context.to_owned(),
                        &matrix_size,
                        &element_list,
                        &First::<$value_type>::new(),
                    )
                    .unwrap();

                    let mut product_vector =
                        SparseVector::<$value_type>::new(&context, matrix_size.row_height_ref()).unwrap();

                    let reducer = MonoidReducer::new(
                    );

                    reducer.to_column_vector(&MonoidPlus::<$value_type>::new(), &matrix, &Assignment::<$value_type>::new(), &mut product_vector, &SelectEntireVector::new(&context), &OperatorOptions::new_default()).unwrap();

                    println!("{}", product_vector);

                    assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
                    assert_eq!(product_vector.element_value_or_default(&1).unwrap(), 2 as $value_type);
                    assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 2 as $value_type);
                    assert_eq!(product_vector.element_value(&9).unwrap(), None);

                    let mask_element_list = VectorElementList::<$value_type>::from_element_vector(vec![
                        (1, 1 as $value_type).into(),
                        (2, 2 as $value_type).into(),
                        (4, 4 as $value_type).into(),
                        // (5, 5).into(),
                    ]);

                    let mask = SparseVector::<$value_type>::from_element_list(
                        &context.to_owned(),
                        matrix_size.row_height_ref(),
                        &mask_element_list,
                        &First::<$value_type>::new(),
                    )
                    .unwrap();

                    let mut product_vector =
                        SparseVector::<$value_type>::new(&context, matrix_size.row_height_ref()).unwrap();

                    reducer
                        .to_column_vector(&MonoidPlus::<$value_type>::new(), &matrix, &Assignment::<$value_type>::new(), &mut product_vector, &mask, &OperatorOptions::new_default())
                        .unwrap();

                    println!("{}", matrix);
                    println!("{}", product_vector);

                    assert_eq!(product_vector.number_of_stored_elements().unwrap(), 3);
                    assert_eq!(product_vector.element_value_or_default(&1).unwrap(), 2 as $value_type);
                    assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 2 as $value_type);
                    assert_eq!(product_vector.element_value(&5).unwrap(), None);
                    assert_eq!(product_vector.element_value(&9).unwrap(), None);
                }

                #[test]
                fn [<test_monoid_to_scalar_reducer_for_matrix_ $value_type>]() {
                    let context = Context::init_ready(Mode::NonBlocking).unwrap();

                    let element_list = MatrixElementList::<$value_type>::from_element_vector(vec![
                        (1, 1, 1 as $value_type).into(),
                        (1, 5, 1 as $value_type).into(),
                        (2, 1, 2 as $value_type).into(),
                        (4, 2, 4 as $value_type).into(),
                        (5, 2, 5 as $value_type).into(),
                    ]);

                    let matrix_size: Size = (10, 15).into();
                    let matrix = SparseMatrix::<$value_type>::from_element_list(
                        &context.to_owned(),
                        &matrix_size,
                        &element_list,
                        &First::<$value_type>::new(),
                    )
                    .unwrap();

                    let mut product = 1 as $value_type;

                    let reducer = MonoidReducer::new(
                    );

                    reducer.matrix_to_scalar(&MonoidPlus::<$value_type>::new(), &matrix, &Assignment::new(), &mut product, &OperatorOptions::new_default(),).unwrap();

                    println!("{}", product);

                    assert_eq!(product, 13 as $value_type);
                }

                #[test]
                fn [<test_monoid_to_scalar_reducer_for_vector_ $value_type>]() {
                    let context = Context::init_ready(Mode::NonBlocking).unwrap();

                    let element_list = VectorElementList::<$value_type>::from_element_vector(vec![
                        (1, 1 as $value_type).into(),
                        (2, 2 as $value_type).into(),
                        (4, 4 as $value_type).into(),
                        (5, 5 as $value_type).into(),
                    ]);

                    let vector_length = 10;
                    let vector = SparseVector::<$value_type>::from_element_list(
                        &context.to_owned(),
                        &vector_length,
                        &element_list,
                        &First::<$value_type>::new(),
                    )
                    .unwrap();

                    let mut product = 0 as $value_type;

                    let reducer = MonoidReducer::new(
                    );

                    reducer.vector_to_scalar(&MonoidPlus::<$value_type>::new(), &vector, &Assignment::new(), &mut product, &OperatorOptions::new_default(),).unwrap();

                    println!("{}", product);

                    assert_eq!(product, 12 as $value_type);
                }
            }
        };
    }

    implement_macro_for_all_value_types_except_bool!(test_monoid);
}
