use std::marker::PhantomData;

use crate::graphblas_bindings::*;
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
    ($monoid_operator_name:ident, $graphblas_operator_trait_name:ident) => {
        pub trait $graphblas_operator_trait_name<T: ValueType> {
            fn graphblas_type() -> GrB_Monoid;
        }

        impl<T: ValueType + $graphblas_operator_trait_name<T>> Monoid<T>
            for $monoid_operator_name<T>
        {
            fn graphblas_type(&self) -> GrB_Monoid {
                T::graphblas_type()
            }
        }

        impl<T: ValueType> $monoid_operator_name<T> {
            pub fn new() -> Self {
                Self {
                    _value_type: PhantomData,
                }
            }
        }
    };
}

macro_rules! implement_typed_monoid_operator {
    ($operator_trait_name:ident,
        $graphblas_operator_name:ident,
        $value_type:ty
    ) => {
        impl $operator_trait_name<$value_type> for $value_type {
            fn graphblas_type() -> GrB_Monoid {
                unsafe { $graphblas_operator_name }
            }
        }
    };
}

#[derive(Debug, Clone)]
pub struct Min<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_monoid_operator!(Min, MinMonoidTyped);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool!(
    implement_typed_monoid_operator,
    MinMonoidTyped,
    GrB_MIN_MONOID
);

#[derive(Debug, Clone)]
pub struct Max<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_monoid_operator!(Max, MaxMonoidTyped);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool!(
    implement_typed_monoid_operator,
    MaxMonoidTyped,
    GrB_MAX_MONOID
);

#[derive(Debug, Clone)]
pub struct Plus<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_monoid_operator!(Plus, PlusMonoidTyped);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool!(
    implement_typed_monoid_operator,
    PlusMonoidTyped,
    GrB_PLUS_MONOID
);

#[derive(Debug, Clone)]
pub struct Times<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_monoid_operator!(Times, TimesMonoidTyped);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_except_bool!(
    implement_typed_monoid_operator,
    TimesMonoidTyped,
    GrB_TIMES_MONOID
);

#[derive(Debug, Clone)]
pub struct Any<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_monoid_operator!(Any, AnyMonoidTyped);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_with_postfix!(
    implement_typed_monoid_operator,
    AnyMonoidTyped,
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

implement_monoid_operator!(LogicalOr, LogicalOrMonoidTyped);
implement_monoid_operator!(LogicalAnd, LogicalAndMonoidTyped);
implement_monoid_operator!(LogicalExclusiveOr, LogicalExclusiveOrMonoidTyped);
implement_monoid_operator!(Equal, EqualMonoidTyped);

implement_typed_monoid_operator!(LogicalOrMonoidTyped, GrB_LOR_MONOID_BOOL, bool);
implement_typed_monoid_operator!(LogicalAndMonoidTyped, GrB_LAND_MONOID_BOOL, bool);
implement_typed_monoid_operator!(LogicalExclusiveOrMonoidTyped, GrB_LXOR_MONOID_BOOL, bool);
implement_typed_monoid_operator!(EqualMonoidTyped, GrB_LXNOR_MONOID_BOOL, bool);

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

implement_monoid_operator!(BitwiseLogicalOr, BitwiseLogicalOrMonoidTyped);
implement_monoid_operator!(BitwiseLogicalAnd, BitwiseLogicalAndMonoidTyped);
implement_monoid_operator!(
    BitwiseLogicalExclusiveOr,
    BitwiseLogicalExclusiveOrMonoidTyped
);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_unsigned_integers_with_postfix!(
    implement_typed_monoid_operator,
    BitwiseLogicalOrMonoidTyped,
    GxB_BOR,
    MONOID
);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_unsigned_integers_with_postfix!(
    implement_typed_monoid_operator,
    BitwiseLogicalAndMonoidTyped,
    GxB_BAND,
    MONOID
);

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_unsigned_integers_with_postfix!(
    implement_typed_monoid_operator,
    BitwiseLogicalExclusiveOrMonoidTyped,
    GxB_BXOR,
    MONOID
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_vector::operations::{
        FromVectorElementList, GetSparseVectorElementList, GetSparseVectorElementValue,
    };
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::context::Context;
    use crate::operators::binary_operator::{Assignment, First};
    use crate::operators::element_wise_addition::{
        ApplyElementWiseVectorAdditionMonoidOperator, ElementWiseVectorAdditionMonoidOperator,
    };
    use crate::operators::mask::SelectEntireVector;
    use crate::operators::options::OptionsForOperatorWithMatrixArguments;

    #[test]
    fn new_binary_operator() {
        let min_monoid = Min::<f32>::new();
        let _graphblas_type = min_monoid.graphblas_type();
    }

    #[test]
    fn test_element_wise_addition_with_equality_operator() {
        let context = Context::init_default().unwrap();

        let operator = Equal::<bool>::new();
        let options = OptionsForOperatorWithMatrixArguments::new_default();
        let equality_operator = ElementWiseVectorAdditionMonoidOperator::new();

        let length = 7;

        let multiplier = SparseVector::<bool>::new(context.clone(), length).unwrap();
        let multiplicant = multiplier.clone();
        let mut product = multiplier.clone();

        // Test multiplication of empty matrices
        equality_operator
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireVector::new(context.clone()),
                &options,
            )
            .unwrap();
        let element_list = product.element_list().unwrap();

        assert_eq!(element_list.length(), 0);

        let multiplier_element_list = VectorElementList::<bool>::from_element_vector(vec![
            (1, false).into(),
            (3, true).into(),
            (5, false).into(),
            (6, true).into(),
        ]);
        let multiplier = SparseVector::<bool>::from_element_list(
            context.clone(),
            length,
            multiplier_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let multiplicant_element_list = VectorElementList::<bool>::from_element_vector(vec![
            (3, true).into(),
            (4, true).into(),
            (5, false).into(),
            (6, false).into(),
        ]);
        let multiplicant = SparseVector::<bool>::from_element_list(
            context.clone(),
            length,
            multiplicant_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        equality_operator
            .apply(
                &multiplier,
                &operator,
                &multiplicant,
                &Assignment::new(),
                &mut product,
                &SelectEntireVector::new(context.clone()),
                &options,
            )
            .unwrap();

        assert_eq!(product.element_value_or_default(0).unwrap(), false); // operator does not apply on empty
        assert_eq!(product.element_value_or_default(1).unwrap(), false); // false unequal to empty
        assert_eq!(product.element_value_or_default(2).unwrap(), false); // operator does not apply on empty
        assert_eq!(product.element_value_or_default(3).unwrap(), true);
        assert_eq!(product.element_value_or_default(4).unwrap(), true); // true and empty => true
        assert_eq!(product.element_value_or_default(5).unwrap(), true); // true and false => true
        assert_eq!(product.element_value_or_default(6).unwrap(), false); // false and true => false
    }
}
