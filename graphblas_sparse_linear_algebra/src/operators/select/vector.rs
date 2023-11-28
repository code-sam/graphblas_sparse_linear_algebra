use suitesparse_graphblas_sys::{
    GrB_Vector_select_BOOL, GrB_Vector_select_FP32, GrB_Vector_select_FP64,
    GrB_Vector_select_INT16, GrB_Vector_select_INT32, GrB_Vector_select_INT64,
    GrB_Vector_select_INT8, GrB_Vector_select_UINT16, GrB_Vector_select_UINT32,
    GrB_Vector_select_UINT64, GrB_Vector_select_UINT8,
};

use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{CallGraphBlasContext, GetContext};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::operators::mask::VectorMask;
use crate::operators::options::OperatorOptions;
use crate::operators::options::OperatorOptionsTrait;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::{ConvertScalar, ValueType};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for VectorSelector {}
unsafe impl Sync for VectorSelector {}

#[derive(Debug, Clone)]
pub struct VectorSelector {}

impl VectorSelector {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait SelectFromVector<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        selector: &impl IndexUnaryOperator<EvaluationDomain>,
        selector_argument: &EvaluationDomain,
        argument: &(impl GetGraphblasSparseVector + GetContext),
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
        product: &mut (impl GetGraphblasSparseVector + GetContext),
        mask: &(impl VectorMask + GetContext),
        options: &OperatorOptions,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_select_from_vector {
    ($selector_argument_type:ty, $_graphblas_implementation_type:ty, $graphblas_operator:ident) => {
        impl SelectFromVector<$selector_argument_type> for VectorSelector {
            fn apply(
                &self,
                selector: &impl IndexUnaryOperator<$selector_argument_type>,
                selector_argument: &$selector_argument_type,
                argument: &(impl GetGraphblasSparseVector + GetContext),
                accumulator: &impl AccumulatorBinaryOperator<$selector_argument_type>,
                product: &mut (impl GetGraphblasSparseVector + GetContext),
                mask: &(impl VectorMask + GetContext),
                options: &OperatorOptions,
            ) -> Result<(), SparseLinearAlgebraError> {
                let selector_argument = selector_argument.to_owned().to_type()?;
                argument.context_ref().call(
                    || unsafe {
                        $graphblas_operator(
                            product.graphblas_vector(),
                            mask.graphblas_vector(),
                            accumulator.accumulator_graphblas_type(),
                            selector.graphblas_type(),
                            argument.graphblas_vector(),
                            selector_argument,
                            options.to_graphblas_descriptor(),
                        )
                    },
                    unsafe { product.graphblas_vector_ref() },
                )?;

                Ok(())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_select_from_vector,
    GrB_Vector_select
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_vector::operations::{
        FromVectorElementList, GetVectorElementValue,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};

    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::operators::index_unary_operator::{IsValueGreaterThan, IsValueLessThan};
    use crate::operators::mask::SelectEntireVector;

    #[test]
    fn test_zero_scalar_selector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (0, 1).into(),
            (1, 2).into(),
            (2, 3).into(),
            (3, 4).into(),
        ]);

        let vector_length: usize = 4;
        let vector = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let index_operator = IsValueGreaterThan::<u8>::new();
        let selector = VectorSelector::new();

        selector
            .apply(
                &index_operator,
                &0,
                &vector,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.element_value_or_default(&0).unwrap(), 1);
        assert_eq!(product_vector.element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 3);
        assert_eq!(product_vector.element_value_or_default(&3).unwrap(), 4);

        let index_operator = IsValueLessThan::<u8>::new();
        let selector = VectorSelector::new();
        selector
            .apply(
                &index_operator,
                &0,
                &vector,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 0);
        assert_eq!(product_vector.element_value(&0).unwrap(), None);
    }

    #[test]
    fn test_scalar_vector_selector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (0, 1).into(),
            (1, 2).into(),
            (2, 3).into(),
            (3, 4).into(),
        ]);

        let vector_length: usize = 4;
        let vector = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let index_operator = IsValueGreaterThan::<u8>::new();
        let selector = VectorSelector::new();

        selector
            .apply(
                &index_operator,
                &1,
                &vector,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_vector.element_value(&0).unwrap(), None);
        assert_eq!(product_vector.element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.element_value_or_default(&2).unwrap(), 3);
        assert_eq!(product_vector.element_value_or_default(&3).unwrap(), 4);

        let index_operator = IsValueLessThan::<u8>::new();
        let selector = VectorSelector::new();
        selector
            .apply(
                &index_operator,
                &3,
                &vector,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(product_vector.element_value_or_default(&0).unwrap(), 1);
        assert_eq!(product_vector.element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.element_value(&2).unwrap(), None);
        assert_eq!(product_vector.element_value(&3).unwrap(), None);
    }
}
