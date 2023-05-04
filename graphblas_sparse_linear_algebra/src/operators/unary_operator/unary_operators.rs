use std::marker::PhantomData;

use crate::bindings_to_graphblas_implementation::*;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types,
};
use crate::value_type::ValueType;

pub trait UnaryOperator<Argument, Product, EvaluationDomain>
where
    Argument: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
{
    fn graphblas_type(&self) -> GrB_UnaryOp;
}

macro_rules! implement_unary_operator {
    ($operator_name:ident,
        $graphblas_operator_name:ident,
        $evaluation_domain:ty
    ) => {
        impl<Argument: ValueType, Product: ValueType>
            UnaryOperator<Argument, Product, $evaluation_domain>
            for $operator_name<Argument, Product, $evaluation_domain>
        {
            fn graphblas_type(&self) -> GrB_UnaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl<Argument: ValueType, Product: ValueType>
            $operator_name<Argument, Product, $evaluation_domain>
        {
            pub fn new() -> Self {
                Self {
                    _argument_type: PhantomData,
                    _product_type: PhantomData,
                    _evaluation_domain: PhantomData,
                }
            }
        }
    };
}

macro_rules! define_unary_operator {
    ($identifier: ident) => {
        #[derive(Debug, Clone)]
        pub struct $identifier<Argument: ValueType, Product: ValueType, EvaluationDomain: ValueType>
        {
            _argument_type: PhantomData<Argument>,
            _product_type: PhantomData<Product>,
            _evaluation_domain: PhantomData<EvaluationDomain>,
        }
    };
}

// z = 1
define_unary_operator!(One);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_unary_operator,
    One,
    GxB_ONE
);

// z = x
define_unary_operator!(Identity);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_unary_operator,
    Identity,
    GrB_IDENTITY
);

//z = -x
define_unary_operator!(AdditiveInverse);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_unary_operator,
    AdditiveInverse,
    GrB_AINV
);

//z = 1/x
define_unary_operator!(MultiplicativeInverse);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_unary_operator,
    MultiplicativeInverse,
    GrB_MINV
);

// z = !x
define_unary_operator!(LogicalNegation);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_unary_operator,
    LogicalNegation,
    GxB_LNOT
);

define_unary_operator!(BitwiseNegation);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_unary_operator,
    BitwiseNegation,
    GrB_BNOT
);

define_unary_operator!(RowIndex);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_unary_operator,
    RowIndex,
    GxB_POSITIONI
);

define_unary_operator!(ColumnIndex);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_unary_operator,
    ColumnIndex,
    GxB_POSITIONJ
);

define_unary_operator!(SquareRoot);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    SquareRoot,
    GxB_SQRT
);

define_unary_operator!(NaturalLogarithm);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    NaturalLogarithm,
    GxB_LOG
);

define_unary_operator!(NaturalExponent);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    NaturalExponent,
    GxB_EXP
);

define_unary_operator!(Base10Logarithm);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Base10Logarithm,
    GxB_LOG10
);

define_unary_operator!(Base2Logarithm);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Base2Logarithm,
    GxB_LOG2
);

define_unary_operator!(Base2Exponent);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Base2Exponent,
    GxB_EXP2
);

// z = exp(x)-1
define_unary_operator!(NaturalExponentMinus1);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    NaturalExponentMinus1,
    GxB_EXPM1
);

// z = log_e (x+1)
define_unary_operator!(NaturalLogarithmOfArgumentPlusOne);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    NaturalLogarithmOfArgumentPlusOne,
    GxB_LOG1P
);

define_unary_operator!(Sine);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Sine,
    GxB_SIN
);

define_unary_operator!(Cosine);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Cosine,
    GxB_COS
);

define_unary_operator!(Tangent);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Tangent,
    GxB_TAN
);

define_unary_operator!(InverseSine);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    InverseSine,
    GxB_ASIN
);

define_unary_operator!(InverseCosine);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    InverseCosine,
    GxB_ACOS
);

define_unary_operator!(InverseTangent);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    InverseTangent,
    GxB_ATAN
);

define_unary_operator!(HyberbolicSine);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    HyberbolicSine,
    GxB_SINH
);

define_unary_operator!(HyberbolicCosine);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    HyberbolicCosine,
    GxB_COSH
);

define_unary_operator!(HyberbolicTangent);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    HyberbolicTangent,
    GxB_TANH
);

define_unary_operator!(InverseHyberbolicSine);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    InverseHyberbolicSine,
    GxB_ASINH
);

define_unary_operator!(InverseHyberbolicCosine);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    InverseHyberbolicCosine,
    GxB_ACOSH
);

define_unary_operator!(InverseHyberbolicTangent);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    InverseHyberbolicTangent,
    GxB_ATANH
);

define_unary_operator!(Sign);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Sign,
    GxB_SIGNUM
);

define_unary_operator!(Ceiling);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Ceiling,
    GxB_CEIL
);

define_unary_operator!(Floor);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Floor,
    GxB_FLOOR
);

define_unary_operator!(Round);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Round,
    GxB_ROUND
);

define_unary_operator!(Truncate);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    Truncate,
    GxB_TRUNC
);

define_unary_operator!(IsInfinite);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    IsInfinite,
    GxB_ISINF
);

define_unary_operator!(IsNaN);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    IsNaN,
    GxB_ISNAN
);

define_unary_operator!(IsFinite);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    IsFinite,
    GxB_ISFINITE
);

define_unary_operator!(NaturalLogarithmOfGammaFunction);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    NaturalLogarithmOfGammaFunction,
    GxB_LGAMMA
);

define_unary_operator!(GammaFunction);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    GammaFunction,
    GxB_TGAMMA
);

define_unary_operator!(ErrorFunction);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    ErrorFunction,
    GxB_ERF
);

define_unary_operator!(ComplimentoryErrorFunction);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    ComplimentoryErrorFunction,
    GxB_ERFC
);

define_unary_operator!(CubeRoot);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    CubeRoot,
    GxB_CBRT
);

define_unary_operator!(NormalisedFraction);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    NormalisedFraction,
    GxB_FREXPX
);

define_unary_operator!(NormalisedExponent);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_unary_operator,
    NormalisedExponent,
    GxB_FREXPE
);

#[cfg(test)]
mod tests {
    use crate::{
        collections::{
            sparse_vector::{
                FromVectorElementList, GetVectorElement, GetVectorElementValue, SparseVector,
                VectorElementList,
            },
            Collection,
        },
        context::{Context, Mode},
        operators::{
            apply::{ApplyUnaryOperator, UnaryOperatorApplier},
            binary_operator::{Assignment, First},
            options::OperatorOptions,
        },
    };

    use super::*;

    #[test]
    fn new_binary_operator() {
        let min_monoid = AdditiveInverse::<f32, f32, f32>::new();
        let _graphblas_type = min_monoid.graphblas_type();
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
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<i64, i64, i64, i64>::new(),
        )
        .unwrap();
        let operator = UnaryOperatorApplier::new(
            &IsFinite::<i64, u8, f32>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        let mut product = SparseVector::new(&context, &vector_length).unwrap();

        operator.apply_to_vector(&vector, &mut product).unwrap();

        println!("{}", product);

        assert_eq!(product.get_element_value_or_default(&6).unwrap(), 1u8);
    }
}
