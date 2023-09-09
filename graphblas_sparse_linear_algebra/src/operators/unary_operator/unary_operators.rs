use std::marker::PhantomData;

use crate::bindings_to_graphblas_implementation::*;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types,
};
use crate::value_type::ValueType;

pub trait UnaryOperator<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn graphblas_type(&self) -> GrB_UnaryOp;
}

// macro_rules! implement_unary_operator {
//     ($operator_name:ident,
//         $graphblas_operator_name:ident,
//         $evaluation_domain:ty
//     ) => {
//         impl UnaryOperator<$evaluation_domain> for $operator_name<$evaluation_domain> {
//             fn graphblas_type(&self) -> GrB_UnaryOp {
//                 unsafe { $graphblas_operator_name }
//             }
//         }

//         impl $operator_name<$evaluation_domain> {
//             pub fn new() -> Self {
//                 Self {
//                     _evaluation_domain: PhantomData,
//                 }
//             }
//         }
//     };
// }

macro_rules! define_unary_operator {
    ($identifier: ident) => {
        #[derive(Debug, Clone)]
        pub struct $identifier<EvaluationDomain: ValueType> {
            _evaluation_domain: PhantomData<EvaluationDomain>,
        }
    };
}

macro_rules! implement_unary_operator {
    ($operator_name:ident, $graphblas_operator_trait_name:ident) => {
        pub trait $graphblas_operator_trait_name<T: ValueType> {
            fn graphblas_type() -> GrB_UnaryOp;
        }

        impl<T: ValueType + $graphblas_operator_trait_name<T>> UnaryOperator<T>
            for $operator_name<T>
        {
            fn graphblas_type(&self) -> GrB_UnaryOp {
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

macro_rules! implement_typed_unary_operator {
    ($operator_trait_name:ident,
        $graphblas_operator_name:ident,
        $value_type:ty
    ) => {
        impl $operator_trait_name<$value_type> for $value_type {
            fn graphblas_type() -> GrB_UnaryOp {
                unsafe { $graphblas_operator_name }
            }
        }
    };
}

// z = 1
define_unary_operator!(One);
implement_unary_operator!(One, OneUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_unary_operator,
    OneUnaryOperatorTyped,
    GxB_ONE
);

// z = x
define_unary_operator!(Identity);
implement_unary_operator!(Identity, IdentityUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_unary_operator,
    IdentityUnaryOperatorTyped,
    GrB_IDENTITY
);

//z = -x
define_unary_operator!(AdditiveInverse);
implement_unary_operator!(AdditiveInverse, AdditiveInverseUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_unary_operator,
    AdditiveInverseUnaryOperatorTyped,
    GrB_AINV
);

//z = 1/x
define_unary_operator!(MultiplicativeInverse);
implement_unary_operator!(MultiplicativeInverse, MultiplicativeInverseUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_unary_operator,
    MultiplicativeInverseUnaryOperatorTyped,
    GrB_MINV
);

// z = !x
define_unary_operator!(LogicalNegation);
implement_unary_operator!(LogicalNegation, LogicalNegationUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_unary_operator,
    LogicalNegationUnaryOperatorTyped,
    GxB_LNOT
);

define_unary_operator!(BitwiseNegation);
implement_unary_operator!(BitwiseNegation, BitwiseNegationUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_unary_operator,
    BitwiseNegationUnaryOperatorTyped,
    GrB_BNOT
);

define_unary_operator!(RowIndex);
implement_unary_operator!(RowIndex, RowIndexUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_typed_unary_operator,
    RowIndexUnaryOperatorTyped,
    GxB_POSITIONI
);

define_unary_operator!(ColumnIndex);
implement_unary_operator!(ColumnIndex, ColumnIndexUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_typed_unary_operator,
    ColumnIndexUnaryOperatorTyped,
    GxB_POSITIONJ
);

define_unary_operator!(SquareRoot);
implement_unary_operator!(SquareRoot, SquareRootUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    SquareRootUnaryOperatorTyped,
    GxB_SQRT
);

define_unary_operator!(NaturalLogarithm);
implement_unary_operator!(NaturalLogarithm, NaturalLogarithmUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    NaturalLogarithmUnaryOperatorTyped,
    GxB_LOG
);

define_unary_operator!(NaturalExponent);
implement_unary_operator!(NaturalExponent, NaturalExponentUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    NaturalExponentUnaryOperatorTyped,
    GxB_EXP
);

define_unary_operator!(Base10Logarithm);
implement_unary_operator!(Base10Logarithm, Base10LogarithmUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    Base10LogarithmUnaryOperatorTyped,
    GxB_LOG10
);

define_unary_operator!(Base2Logarithm);
implement_unary_operator!(Base2Logarithm, Base2LogarithmUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    Base2LogarithmUnaryOperatorTyped,
    GxB_LOG2
);

define_unary_operator!(Base2Exponent);
implement_unary_operator!(Base2Exponent, Base2ExponentUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    Base2ExponentUnaryOperatorTyped,
    GxB_EXP2
);

// z = exp(x)-1
define_unary_operator!(NaturalExponentMinus1);
implement_unary_operator!(NaturalExponentMinus1, NaturalExponentMinus1UnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    NaturalExponentMinus1UnaryOperatorTyped,
    GxB_EXPM1
);

// z = log_e (x+1)
define_unary_operator!(NaturalLogarithmOfArgumentPlusOne);
implement_unary_operator!(NaturalLogarithmOfArgumentPlusOne, NaturalLogarithmOfArgumentPlusOneUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    NaturalLogarithmOfArgumentPlusOneUnaryOperatorTyped,
    GxB_LOG1P
);

define_unary_operator!(Sine);
implement_unary_operator!(Sine, SineUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    SineUnaryOperatorTyped,
    GxB_SIN
);

define_unary_operator!(Cosine);
implement_unary_operator!(Cosine, CosineUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    CosineUnaryOperatorTyped,
    GxB_COS
);

define_unary_operator!(Tangent);
implement_unary_operator!(Tangent, TangentUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    TangentUnaryOperatorTyped,
    GxB_TAN
);

define_unary_operator!(InverseSine);
implement_unary_operator!(InverseSine, InverseSineUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    InverseSineUnaryOperatorTyped,
    GxB_ASIN
);

define_unary_operator!(InverseCosine);
implement_unary_operator!(InverseCosine, InverseCosineUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    InverseCosineUnaryOperatorTyped,
    GxB_ACOS
);

define_unary_operator!(InverseTangent);
implement_unary_operator!(InverseTangent, InverseTangentUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    InverseTangentUnaryOperatorTyped,
    GxB_ATAN
);

define_unary_operator!(HyberbolicSine);
implement_unary_operator!(HyberbolicSine, HyberbolicSineUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    HyberbolicSineUnaryOperatorTyped,
    GxB_SINH
);

define_unary_operator!(HyberbolicCosine);
implement_unary_operator!(HyberbolicCosine, HyberbolicCosineUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    HyberbolicCosineUnaryOperatorTyped,
    GxB_COSH
);

define_unary_operator!(HyberbolicTangent);
implement_unary_operator!(HyberbolicTangent, HyberbolicTangentUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    HyberbolicTangentUnaryOperatorTyped,
    GxB_TANH
);

define_unary_operator!(InverseHyberbolicSine);
implement_unary_operator!(InverseHyberbolicSine, InverseHyberbolicSineUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    InverseHyberbolicSineUnaryOperatorTyped,
    GxB_ASINH
);

define_unary_operator!(InverseHyberbolicCosine);
implement_unary_operator!(InverseHyberbolicCosine, InverseHyberbolicCosineUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    InverseHyberbolicCosineUnaryOperatorTyped,
    GxB_ACOSH
);

define_unary_operator!(InverseHyberbolicTangent);
implement_unary_operator!(InverseHyberbolicTangent, InverseHyberbolicTangentUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    InverseHyberbolicTangentUnaryOperatorTyped,
    GxB_ATANH
);

define_unary_operator!(Sign);
implement_unary_operator!(Sign, SignUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    SignUnaryOperatorTyped,
    GxB_SIGNUM
);

define_unary_operator!(Ceiling);
implement_unary_operator!(Ceiling, CeilingUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    CeilingUnaryOperatorTyped,
    GxB_CEIL
);

define_unary_operator!(Floor);
implement_unary_operator!(Floor, FloorUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    FloorUnaryOperatorTyped,
    GxB_FLOOR
);

define_unary_operator!(Round);
implement_unary_operator!(Round, RoundUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    RoundUnaryOperatorTyped,
    GxB_ROUND
);

define_unary_operator!(Truncate);
implement_unary_operator!(Truncate, TruncateUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    TruncateUnaryOperatorTyped,
    GxB_TRUNC
);

define_unary_operator!(IsInfinite);
implement_unary_operator!(IsInfinite, IsInfiniteUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    IsInfiniteUnaryOperatorTyped,
    GxB_ISINF
);

define_unary_operator!(IsNaN);
implement_unary_operator!(IsNaN, IsNaNUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    IsNaNUnaryOperatorTyped,
    GxB_ISNAN
);

define_unary_operator!(IsFinite);
implement_unary_operator!(IsFinite, IsFiniteUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    IsFiniteUnaryOperatorTyped,
    GxB_ISFINITE
);

define_unary_operator!(NaturalLogarithmOfGammaFunction);
implement_unary_operator!(NaturalLogarithmOfGammaFunction, NaturalLogarithmOfGammaFunctionUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    NaturalLogarithmOfGammaFunctionUnaryOperatorTyped,
    GxB_LGAMMA
);

define_unary_operator!(GammaFunction);
implement_unary_operator!(GammaFunction, GammaFunctionUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    GammaFunctionUnaryOperatorTyped,
    GxB_TGAMMA
);

define_unary_operator!(ErrorFunction);
implement_unary_operator!(ErrorFunction, ErrorFunctionUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    ErrorFunctionUnaryOperatorTyped,
    GxB_ERF
);

define_unary_operator!(ComplimentoryErrorFunction);
implement_unary_operator!(ComplimentoryErrorFunction, ComplimentoryErrorFunctionUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    ComplimentoryErrorFunctionUnaryOperatorTyped,
    GxB_ERFC
);

define_unary_operator!(CubeRoot);
implement_unary_operator!(CubeRoot, CubeRootUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    CubeRootUnaryOperatorTyped,
    GxB_CBRT
);

define_unary_operator!(NormalisedFraction);
implement_unary_operator!(NormalisedFraction, NormalisedFractionUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    NormalisedFractionUnaryOperatorTyped,
    GxB_FREXPX
);

define_unary_operator!(NormalisedExponent);
implement_unary_operator!(NormalisedExponent, NormalisedExponentUnaryOperatorTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_unary_operator,
    NormalisedExponentUnaryOperatorTyped,
    GxB_FREXPE
);

#[cfg(test)]
mod tests {
    use crate::{
        collections::{
            sparse_vector::{
                operations::GetVectorElementValue, FromVectorElementList, SparseVector,
                VectorElementList,
            },
            Collection,
        },
        context::{Context, Mode},
        operators::{
            apply::{ApplyUnaryOperator, UnaryOperatorApplier},
            binary_operator::{Assignment, First},
            mask::SelectEntireVector,
            options::OperatorOptions,
        },
    };

    use super::*;

    #[test]
    fn new_binary_operator() {
        let additive_inverse = AdditiveInverse::<f32>::new();
        let _graphblas_type = additive_inverse.graphblas_type();
    }

    #[test]
    fn test_is_finite_and_type_casting() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<i64>::from_element_vector(vec![
            (1, 1).into(),
            (3, 2).into(),
            (6, -3).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<i64>::from_element_list(
            &context.to_owned(),
            &vector_length,
            &element_list,
            &First::<i64>::new(),
        )
        .unwrap();
        let operator = UnaryOperatorApplier::new();

        let mut product = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        operator
            .apply_to_vector(
                &IsFinite::<f32>::new(),
                &vector,
                &Assignment::new(),
                &mut product,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product);

        assert_eq!(product.get_element_value_or_default(&6).unwrap(), 1u8);
    }
}
