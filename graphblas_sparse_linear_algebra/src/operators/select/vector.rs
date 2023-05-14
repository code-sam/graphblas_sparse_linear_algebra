use std::ptr;

use std::marker::PhantomData;

use suitesparse_graphblas_sys::{
    GrB_IndexUnaryOp, GrB_Vector_select_BOOL, GrB_Vector_select_FP32, GrB_Vector_select_FP64,
    GrB_Vector_select_INT16, GrB_Vector_select_INT32, GrB_Vector_select_INT64,
    GrB_Vector_select_INT8, GrB_Vector_select_UINT16, GrB_Vector_select_UINT32,
    GrB_Vector_select_UINT64, GrB_Vector_select_UINT8,
};

use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};

use crate::collections::sparse_vector::{GraphblasSparseVectorTrait, SparseVector};
use crate::operators::index_unary_operator::IndexUnaryOperator;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::{AsBoolean, ConvertScalar, ValueType};

use crate::bindings_to_graphblas_implementation::{GrB_BinaryOp, GrB_Descriptor};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<EvaluationDomain: ValueType> Send for VectorSelector<EvaluationDomain> {}
unsafe impl<EvaluationDomain: ValueType> Sync for VectorSelector<EvaluationDomain> {}

#[derive(Debug, Clone)]
pub struct VectorSelector<EvaluationDomain: ValueType> {
    _evaluation_domain: PhantomData<EvaluationDomain>,

    selector: GrB_IndexUnaryOp,
    accumulator: GrB_BinaryOp,
    options: GrB_Descriptor,
}

impl<EvaluationDomain: ValueType> VectorSelector<EvaluationDomain> {
    pub fn new(
        selector: &impl IndexUnaryOperator<EvaluationDomain>,
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<EvaluationDomain>,
    ) -> Self {
        Self {
            selector: selector.graphblas_type(),
            accumulator: accumulator.accumulator_graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _evaluation_domain: PhantomData,
        }
    }
}

pub trait SelectFromVector<EvaluationDomain: ValueType> {
    fn apply(
        &self,
        argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        selector_argument: &EvaluationDomain,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_mask(
        &self,
        argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
        product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
        selector_argument: &EvaluationDomain,
        mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_select_from_vector {
    ($selector_argument_type:ty, $_graphblas_implementation_type:ty, $graphblas_operator:ident) => {
        impl SelectFromVector<$selector_argument_type> for VectorSelector<$selector_argument_type> {
            fn apply(
                &self,
                argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
                product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
                selector_argument: &$selector_argument_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let selector_argument = selector_argument.clone().to_type()?;
                argument.context_ref().call(
                    || unsafe {
                        $graphblas_operator(
                            product.graphblas_vector(),
                            ptr::null_mut(),
                            self.accumulator,
                            self.selector,
                            argument.graphblas_vector(),
                            selector_argument,
                            self.options,
                        )
                    },
                    unsafe { product.graphblas_vector_ref() },
                )?;

                Ok(())
            }

            fn apply_with_mask(
                &self,
                argument: &(impl GraphblasSparseVectorTrait + ContextTrait),
                product: &mut (impl GraphblasSparseVectorTrait + ContextTrait),
                selector_argument: &$selector_argument_type,
                mask: &(impl GraphblasSparseVectorTrait + ContextTrait),
            ) -> Result<(), SparseLinearAlgebraError> {
                let selector_argument = selector_argument.clone().to_type()?;
                argument.context_ref().call(
                    || unsafe {
                        $graphblas_operator(
                            product.graphblas_vector(),
                            mask.graphblas_vector(),
                            self.accumulator,
                            self.selector,
                            argument.graphblas_vector(),
                            selector_argument,
                            self.options,
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

    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};

    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, VectorElementList,
    };
    use crate::operators::index_unary_operator::{IsValueGreaterThan, IsValueLessThan};

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
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let index_operator = IsValueGreaterThan::<u8>::new();
        let selector = VectorSelector::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        selector.apply(&vector, &mut product_vector, &0).unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value_or_default(&0).unwrap(), 1);
        assert_eq!(product_vector.get_element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 3);
        assert_eq!(product_vector.get_element_value_or_default(&3).unwrap(), 4);

        let index_operator = IsValueLessThan::<u8>::new();
        let selector = VectorSelector::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );
        selector.apply(&vector, &mut product_vector, &0).unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 0);
        assert_eq!(product_vector.get_element_value(&0).unwrap(), None);
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
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let index_operator = IsValueGreaterThan::<u8>::new();
        let selector = VectorSelector::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        selector.apply(&vector, &mut product_vector, &1).unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_vector.get_element_value(&0).unwrap(), None);
        assert_eq!(product_vector.get_element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 3);
        assert_eq!(product_vector.get_element_value_or_default(&3).unwrap(), 4);

        let index_operator = IsValueLessThan::<u8>::new();
        let selector = VectorSelector::new(
            &index_operator,
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );
        selector.apply(&vector, &mut product_vector, &3).unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(product_vector.get_element_value_or_default(&0).unwrap(), 1);
        assert_eq!(product_vector.get_element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), None);
        assert_eq!(product_vector.get_element_value(&3).unwrap(), None);
    }
}
