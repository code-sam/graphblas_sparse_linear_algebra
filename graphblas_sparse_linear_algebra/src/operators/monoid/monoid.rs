use std::marker::PhantomData;

use crate::bindings_to_graphblas_implementation::*;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_with_postfix,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_unsigned_integers_with_postfix,
};
use crate::value_type::ValueType;

pub trait Monoid<T>
where
    T: ValueType,
{
    fn graphblas_type(&self) -> GrB_Monoid;
}

macro_rules! implement_monoid_operator {
    ($operator_name:ident,
        $graphblas_operator_name:ident,
        $value_type:ty
    ) => {
        impl Monoid<$value_type> for $operator_name<$value_type> {
            fn graphblas_type(&self) -> GrB_Monoid {
                unsafe { $graphblas_operator_name }
            }
        }

        impl $operator_name<$value_type> {
            pub fn new() -> Self {
                $operator_name {
                    _value_type: PhantomData,
                }
            }
        }
    };
}

#[derive(Debug, Clone)]
pub struct Min<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool!(
    implement_monoid_operator,
    Min,
    GrB_MIN_MONOID
);

#[derive(Debug, Clone)]
pub struct Max<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool!(
    implement_monoid_operator,
    Max,
    GrB_MAX_MONOID
);

#[derive(Debug, Clone)]
pub struct Plus<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool!(
    implement_monoid_operator,
    Plus,
    GrB_PLUS_MONOID
);

#[derive(Debug, Clone)]
pub struct Times<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool!(
    implement_monoid_operator,
    Times,
    GrB_TIMES_MONOID
);

#[derive(Debug, Clone)]
pub struct Any<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_with_postfix!(
    implement_monoid_operator,
    Any,
    GxB_ANY,
    MONOID
);

#[derive(Debug, Clone)]
pub struct LogicalOr<T: ValueType> {
    _value_type: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct LogicalAnd<T: ValueType> {
    _value_type: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct LogicalExclusiveOr<T: ValueType> {
    _value_type: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct Equal<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_monoid_operator!(LogicalOr, GrB_LOR_MONOID_BOOL, bool);
implement_monoid_operator!(LogicalAnd, GrB_LAND_MONOID_BOOL, bool);
implement_monoid_operator!(LogicalExclusiveOr, GrB_LXOR_MONOID_BOOL, bool);
implement_monoid_operator!(Equal, GrB_LXNOR_MONOID_BOOL, bool);

#[derive(Debug, Clone)]
pub struct BitwiseLogicalOr<T: ValueType> {
    _value_type: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct BitwiseLogicalAnd<T: ValueType> {
    _value_type: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct BitwiseLogicalExclusiveOr<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_unsigned_integers_with_postfix!(
    implement_monoid_operator,
    BitwiseLogicalOr,
    GxB_BOR,
    MONOID
);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_unsigned_integers_with_postfix!(
    implement_monoid_operator,
    BitwiseLogicalAnd,
    GxB_BAND,
    MONOID
);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_unsigned_integers_with_postfix!(
    implement_monoid_operator,
    BitwiseLogicalExclusiveOr,
    GxB_BXOR,
    MONOID
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementList, GetVectorElementValue, SparseVector,
        VectorElementList,
    };
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::operators::element_wise_addition::{
        ApplyElementWiseVectorAdditionMonoidOperator, ElementWiseVectorAdditionMonoidOperator,
    };
    use crate::operators::options::OperatorOptions;

    #[test]
    fn new_binary_operator() {
        let min_monoid = Min::<f32>::new();
        let _graphblas_type = min_monoid.graphblas_type();
    }

    #[test]
    fn test_element_wise_addition_with_equality_operator() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let operator = Equal::<bool>::new();
        let options = OperatorOptions::new_default();
        let equality_operator =
            ElementWiseVectorAdditionMonoidOperator::<bool>::new(&operator, &options, None);

        let length = 7;

        let multiplier = SparseVector::<bool>::new(&context, &length).unwrap();
        let multiplicant = multiplier.clone();
        let mut product = multiplier.clone();

        // Test multiplication of empty matrices
        equality_operator
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();
        let element_list = product.get_element_list().unwrap();

        assert_eq!(element_list.length(), 0);

        let multiplier_element_list = VectorElementList::<bool>::from_element_vector(vec![
            (1, false).into(),
            (3, true).into(),
            (5, false).into(),
            (6, true).into(),
        ]);
        let multiplier = SparseVector::<bool>::from_element_list(
            &context,
            &length,
            &multiplier_element_list,
            &First::<bool, bool, bool, bool>::new(),
        )
        .unwrap();

        let multiplicant_element_list = VectorElementList::<bool>::from_element_vector(vec![
            (3, true).into(),
            (4, true).into(),
            (5, false).into(),
            (6, false).into(),
        ]);
        let multiplicant = SparseVector::<bool>::from_element_list(
            &context,
            &length,
            &multiplicant_element_list,
            &First::<bool, bool, bool, bool>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        equality_operator
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value_or_default(&0).unwrap(), false); // operator does not apply on empty
        assert_eq!(product.get_element_value_or_default(&1).unwrap(), false); // false unequal to empty
        assert_eq!(product.get_element_value_or_default(&2).unwrap(), false); // operator does not apply on empty
        assert_eq!(product.get_element_value_or_default(&3).unwrap(), true);
        assert_eq!(product.get_element_value_or_default(&4).unwrap(), true); // true and empty => true
        assert_eq!(product.get_element_value_or_default(&5).unwrap(), true); // true and false => true
        assert_eq!(product.get_element_value_or_default(&6).unwrap(), false); // false and true => false
    }
}
