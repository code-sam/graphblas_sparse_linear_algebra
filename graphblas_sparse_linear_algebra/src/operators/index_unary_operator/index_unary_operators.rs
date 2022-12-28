use std::marker::PhantomData;

use crate::bindings_to_graphblas_implementation::*;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types,
};
use crate::value_type::ValueType;

pub trait IndexUnaryOperator<FirstArgument, SecondArgument, Product, EvaluationDomain>
where
    FirstArgument: ValueType,
    SecondArgument: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
{
    fn graphblas_type(&self) -> GrB_IndexUnaryOp;
}

macro_rules! implement_index_unary_operator {
    ($operator_name:ident,
        $graphblas_operator_name:ident,
        $evaluation_domain:ty
    ) => {
        impl<FirstArgument: ValueType, SecondArgument: ValueType, Product: ValueType>
            IndexUnaryOperator<FirstArgument, SecondArgument, Product, $evaluation_domain>
            for $operator_name<FirstArgument, SecondArgument, Product, $evaluation_domain>
        {
            fn graphblas_type(&self) -> GrB_IndexUnaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl<FirstArgument: ValueType, SecondArgument: ValueType, Product: ValueType>
            $operator_name<FirstArgument, SecondArgument, Product, $evaluation_domain>
        {
            pub fn new() -> Self {
                Self {
                    _first_argument_type: PhantomData,
                    _second_argument_type: PhantomData,
                    _product_type: PhantomData,
                    _evaluation_domain: PhantomData,
                }
            }
        }
    };
}

macro_rules! implement_generic_index_unary_operator {
    ($operator_name:ident,
        $graphblas_operator_name:ident
    ) => {
        impl<
                FirstArgument: ValueType,
                SecondArgument: ValueType,
                Product: ValueType,
                EvaluationDomain: ValueType,
            > IndexUnaryOperator<FirstArgument, SecondArgument, Product, EvaluationDomain>
            for $operator_name<FirstArgument, SecondArgument, Product, EvaluationDomain>
        {
            fn graphblas_type(&self) -> GrB_IndexUnaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl<
                FirstArgument: ValueType,
                SecondArgument: ValueType,
                Product: ValueType,
                EvaluationDomain: ValueType,
            > $operator_name<FirstArgument, SecondArgument, Product, EvaluationDomain>
        {
            pub fn new() -> Self {
                Self {
                    _first_argument_type: PhantomData,
                    _second_argument_type: PhantomData,
                    _product_type: PhantomData,
                    _evaluation_domain: PhantomData,
                }
            }
        }
    };
}

macro_rules! define_index_unary_operator {
    ($identifier: ident) => {
        #[derive(Debug, Clone)]
        pub struct $identifier<
            FirstArgument: ValueType,
            SecondArgument: ValueType,
            Product: ValueType,
            EvaluationDomain: ValueType,
        > {
            _first_argument_type: PhantomData<FirstArgument>,
            _second_argument_type: PhantomData<SecondArgument>,
            _product_type: PhantomData<Product>,
            _evaluation_domain: PhantomData<EvaluationDomain>,
        }
    };
}

// z = i + y
define_index_unary_operator!(PlusRowIndex);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_index_unary_operator,
    PlusRowIndex,
    GrB_ROWINDEX
);

// z = j - i + y
define_index_unary_operator!(PlusDiagonalIndex);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_index_unary_operator,
    PlusDiagonalIndex,
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
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_index_unary_operator,
    IsValueNotEqualTo,
    GrB_VALUENE
);

define_index_unary_operator!(IsValueEqualTo);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_index_unary_operator,
    IsValueEqualTo,
    GrB_VALUEEQ
);

define_index_unary_operator!(IsValueGreaterThan);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_index_unary_operator,
    IsValueGreaterThan,
    GrB_VALUEGT
);

define_index_unary_operator!(IsValueLessThan);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_index_unary_operator,
    IsValueLessThan,
    GrB_VALUELT
);

define_index_unary_operator!(IsValueLessThanOrEqualTo);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_index_unary_operator,
    IsValueLessThanOrEqualTo,
    GrB_VALUELE
);

#[cfg(test)]
mod tests {
    use crate::{
        collections::{
            collection::Collection,
            sparse_vector::{
                FromVectorElementList, GetVectorElement, GetVectorElementValue, SparseVector,
                VectorElementList,
            },
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
    //         &context.clone(),
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
