use std::marker::PhantomData;

use crate::graphblas_bindings::*;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types,
};
use crate::value_type::ValueType;

// TODO: review EvaluationDomain. Typecasting may not work as expected, e.g. for less-than ot greater-than operators.
pub trait IndexUnaryOperator<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn graphblas_type(&self) -> GrB_IndexUnaryOp;
}

macro_rules! implement_index_unary_operator {
    ($operator_name:ident, $graphblas_operator_trait_name:ident) => {
        pub trait $graphblas_operator_trait_name<T: ValueType> {
            fn graphblas_type() -> GrB_IndexUnaryOp;
        }

        impl<T: ValueType + $graphblas_operator_trait_name<T>> IndexUnaryOperator<T>
            for $operator_name<T>
        {
            fn graphblas_type(&self) -> GrB_IndexUnaryOp {
                T::graphblas_type()
            }
        }

        impl<T: ValueType> $operator_name<T> {
            pub fn new() -> Self {
                Self {
                    _evaluation_domain: PhantomData,
                }
            }
        }
    };
}

macro_rules! implement_typed_index_unary_operator {
    ($operator_trait_name:ident,
        $graphblas_operator_name:ident,
        $value_type:ty
    ) => {
        impl $operator_trait_name<$value_type> for $value_type {
            fn graphblas_type() -> GrB_IndexUnaryOp {
                unsafe { $graphblas_operator_name }
            }
        }
    };
}

macro_rules! implement_generic_index_unary_operator {
    ($operator_name:ident,
        $graphblas_operator_name:ident
    ) => {
        impl<EvaluationDomain: ValueType> IndexUnaryOperator<EvaluationDomain>
            for $operator_name<EvaluationDomain>
        {
            fn graphblas_type(&self) -> GrB_IndexUnaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl<EvaluationDomain: ValueType> $operator_name<EvaluationDomain> {
            pub fn new() -> Self {
                Self {
                    _evaluation_domain: PhantomData,
                }
            }
        }
    };
}

macro_rules! define_index_unary_operator {
    ($identifier: ident) => {
        #[derive(Debug, Clone)]
        pub struct $identifier<EvaluationDomain: ValueType> {
            _evaluation_domain: PhantomData<EvaluationDomain>,
        }
    };
}

// z = i + y
define_index_unary_operator!(PlusRowIndex);
implement_index_unary_operator!(PlusRowIndex, PlusRowIndexTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_typed_index_unary_operator,
    PlusRowIndexTyped,
    GrB_ROWINDEX
);

// z = j - i + y
define_index_unary_operator!(PlusDiagonalIndex);
implement_index_unary_operator!(PlusDiagonalIndex, PlusDiagonalIndexTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_typed_index_unary_operator,
    PlusDiagonalIndexTyped,
    GrB_DIAGINDEX
);

// z=(j<=(i+y))
// true for entries on or below the yth diagonal
define_index_unary_operator!(IsOnOrBelowDiagonal);
implement_generic_index_unary_operator!(IsOnOrBelowDiagonal, GrB_TRIL);

// z=(j>=(i+y))
// true for entries on or above the yth diagonal
define_index_unary_operator!(IsOnOrAboveDiagonal);
implement_generic_index_unary_operator!(IsOnOrAboveDiagonal, GrB_TRIU);

// z=(j==(i+y))
// true for entries on the yth diagonal
define_index_unary_operator!(IsOnDiagonal);
implement_generic_index_unary_operator!(IsOnDiagonal, GrB_DIAG);

// z=(j<=y)
// true for entries in columns 0 to y
define_index_unary_operator!(IsUpToAndIncludingColumn);
implement_generic_index_unary_operator!(IsUpToAndIncludingColumn, GrB_COLLE);

// z=(j>y)
// true for entries in columns y+1 and above
define_index_unary_operator!(IsAfterColumn);
implement_generic_index_unary_operator!(IsAfterColumn, GrB_COLGT);

// z=(i<=y)
// true for entries in rows 0 to y
define_index_unary_operator!(IsUpToAndIncludingRow);
implement_generic_index_unary_operator!(IsUpToAndIncludingRow, GrB_ROWLE);

// z=(i>y)
// true for entries in rows y+1 and above
define_index_unary_operator!(IsAfterRow);
implement_generic_index_unary_operator!(IsAfterRow, GrB_ROWGT);

define_index_unary_operator!(IsValueNotEqualTo);
implement_index_unary_operator!(IsValueNotEqualTo, IsValueNotEqualToTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_index_unary_operator,
    IsValueNotEqualToTyped,
    GrB_VALUENE
);

define_index_unary_operator!(IsValueEqualTo);
implement_index_unary_operator!(IsValueEqualTo, IsValueEqualToTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_index_unary_operator,
    IsValueEqualToTyped,
    GrB_VALUEEQ
);

define_index_unary_operator!(IsValueGreaterThan);
implement_index_unary_operator!(IsValueGreaterThan, IsValueGreaterThanTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_index_unary_operator,
    IsValueGreaterThanTyped,
    GrB_VALUEGT
);

define_index_unary_operator!(IsValueLessThan);
implement_index_unary_operator!(IsValueLessThan, IsValueLessThanTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_index_unary_operator,
    IsValueLessThanTyped,
    GrB_VALUELT
);

define_index_unary_operator!(IsValueLessThanOrEqualTo);
implement_index_unary_operator!(IsValueLessThanOrEqualTo, IsValueLessThanOrEqualToTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_index_unary_operator,
    IsValueLessThanOrEqualToTyped,
    GrB_VALUELE
);

#[cfg(test)]
mod tests {
    use crate::{
        collections::{
            sparse_vector::{FromVectorElementList, SparseVector, VectorElementList},
            Collection,
        },
        context::{Context, Mode},
        operators::{
            apply::{ApplyIndexUnaryOperator, UnaryOperatorApplier},
            binary_operator::First,
            options::OperatorOptions,
        },
    };

    use super::*;

    // #[test]
    // fn new_binary_operator() {
    //     let min_monoid = AdditiveInverse::<f32, f32, f32>::new();
    //     let _graphblas_type = min_monoid.graphblas_type();
    // }

    // #[test]
    // fn test_is_finite_and_type_casting() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();

    //     let element_list = VectorElementList::<i64>::from_element_vector(vec![
    //         (1, 1).into(),
    //         (3, 2).into(),
    //         (6, -3).into(),
    //     ]);

    //     let vector_length: usize = 10;
    //     let vector = SparseVector::<i64>::from_element_list(
    //         &context.to_owned(),
    //         &vector_length,
    //         &element_list,
    //         &First::<i64, i64, i64, i64>::new(),
    //     )
    //     .unwrap();
    //     let operator = UnaryOperatorApplier::new(
    //         &IsFinite::<i64, u8, f32>::new(),
    //         &OperatorOptions::new_default(),
    //         None,
    //     );

    //     let mut product = SparseVector::new(&context, &vector_length).unwrap();

    //     operator.apply_to_vector(&vector, &mut product).unwrap();

    //     println!("{}", product);

    //     assert_eq!(product.get_element_value(&6).unwrap(), 1u8);
    // }
}
