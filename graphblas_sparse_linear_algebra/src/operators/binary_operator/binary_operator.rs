use std::marker::PhantomData;

use crate::bindings_to_graphblas_implementation::*;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types,
    implement_macro_with_2_type_trait_and_output_type_and_typed_graphblas_function_for_all_value_types,
};
use crate::value_type::ValueType;

pub trait BinaryOperator<X, Y, Z, T>
where
    X: ValueType, // left-hand-side
    Y: ValueType, // right-hand-side
    Z: ValueType, // result
    T: ValueType,
{
    fn graphblas_type(&self) -> GrB_BinaryOp;
}

pub trait ReturnsBool {}

macro_rules! implement_binary_operator {
    (
        $operator_name:ident,
        $graphblas_operator_name:ident,
        $evaluation_domain: ty
    ) => {
        impl<X: ValueType, Y: ValueType, Z: ValueType> BinaryOperator<X, Y, Z, $evaluation_domain>
            for $operator_name<X, Y, Z, $evaluation_domain>
        {
            fn graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl<X: ValueType, Y: ValueType, Z: ValueType> $operator_name<X, Y, Z, $evaluation_domain> {
            pub fn new() -> Self {
                Self {
                    _value_type_left_input: PhantomData,
                    _value_type_right_input: PhantomData,
                    _value_type_output: PhantomData,
                    _evaluation_domain: PhantomData,
                }
            }
        }
    };
}

macro_rules! implement_binary_operator_with_bool_return_type {
    (
        $operator_name:ident,
        $graphblas_operator_name:ident,
        $evaluation_domain: ty
    ) => {
        impl<X: ValueType, Y: ValueType> BinaryOperator<X, Y, bool, $evaluation_domain>
            for $operator_name<X, Y, bool, $evaluation_domain>
        {
            fn graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl<X: ValueType, Y: ValueType> $operator_name<X, Y, bool, $evaluation_domain> {
            pub fn new() -> Self {
                Self {
                    _value_type_left_input: PhantomData,
                    _value_type_right_input: PhantomData,
                    _value_type_output: PhantomData,
                    _evaluation_domain: PhantomData,
                }
            }
        }

        impl<X: ValueType, Y: ValueType> ReturnsBool
            for $operator_name<X, Y, bool, $evaluation_domain>
        {
        }
    };
}

// x = first(x,y)
#[derive(Debug, Clone)]
pub struct First<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    First,
    GrB_FIRST
);

// y = second(x,y)
#[derive(Debug, Clone)]
pub struct Second<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Second,
    GrB_SECOND
);

// z = 1
#[derive(Debug, Clone)]
pub struct One<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    One,
    GrB_ONEB
);

// z = x^y (z = x.pow(y))
#[derive(Debug, Clone)]
pub struct Power<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Power,
    GxB_POW
);

// z = x+y
#[derive(Debug, Clone)]
pub struct Plus<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Plus,
    GrB_PLUS
);

// z = x-y
#[derive(Debug, Clone)]
pub struct Minus<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Minus,
    GrB_MINUS
);

// z = y-x
#[derive(Debug, Clone)]
pub struct ReverseMinus<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    ReverseMinus,
    GxB_RMINUS
);

// z = x*y
#[derive(Debug, Clone)]
pub struct Times<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Times,
    GrB_TIMES
);

// z = x/y
#[derive(Debug, Clone)]
pub struct Divide<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Divide,
    GrB_DIV
);

// z = x/y
#[derive(Debug, Clone)]
pub struct ReverseDivide<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    ReverseDivide,
    GxB_RDIV
);

// z = x==y
#[derive(Debug, Clone)]
pub struct IsEqual<X, Y, bool, T>
where
    X: ValueType,
    Y: ValueType,
    // Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<bool>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsEqual,
    GrB_EQ
);

// z = x==y
#[derive(Debug, Clone)]
pub struct IsEqualTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsEqualTyped,
    GxB_ISEQ
);

// z = x~!=y
#[derive(Debug, Clone)]
pub struct IsNotEqual<X, Y, bool, T>
where
    X: ValueType,
    Y: ValueType,
    // Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<bool>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsNotEqual,
    GrB_NE
);

// z = x==y
#[derive(Debug, Clone)]
pub struct IsNotEqualTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsNotEqualTyped,
    GxB_ISNE
);

// z = any(x,y), selected according to fastest computation speed
#[derive(Debug, Clone)]
pub struct Any<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Any,
    GxB_ANY
);

// z = min(x,y)
#[derive(Debug, Clone)]
pub struct Min<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Min,
    GrB_MIN
);

// z = max(x,y)
#[derive(Debug, Clone)]
pub struct Max<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Max,
    GrB_MAX
);

// z = (x>y)
#[derive(Debug, Clone)]
pub struct IsGreaterThan<X, Y, bool, T>
where
    X: ValueType,
    Y: ValueType,
    // Z: bool,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<bool>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsGreaterThan,
    GrB_GT
);

// z = (x>y)
#[derive(Debug, Clone)]
pub struct IsGreaterThanTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsGreaterThanTyped,
    GxB_ISGT
);

// z = (x>=y)
#[derive(Debug, Clone)]
pub struct IsGreaterThanOrEqualTo<X, Y, bool, T>
where
    X: ValueType,
    Y: ValueType,
    // Z: bool,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<bool>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsGreaterThanOrEqualTo,
    GrB_GE
);

// z = (x>=y)
#[derive(Debug, Clone)]
pub struct IsGreaterThanOrEqualToTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsGreaterThanOrEqualToTyped,
    GxB_ISGE
);

// z = (x<y)
#[derive(Debug, Clone)]
pub struct IsLessThan<X, Y, bool, T>
where
    X: ValueType,
    Y: ValueType,
    // Z: bool,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<bool>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsLessThan,
    GrB_LT
);

// z = (x<y)
#[derive(Debug, Clone)]
pub struct IsLessThanTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsLessThanTyped,
    GxB_ISLT
);

// z = (x<=y)
#[derive(Debug, Clone)]
pub struct IsLessThanOrEqualTo<X, Y, bool, T>
where
    X: ValueType,
    Y: ValueType,
    // Z: bool,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<bool>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsLessThanOrEqualTo,
    GrB_LE
);

// z = (x<=y)
#[derive(Debug, Clone)]
pub struct IsLessThanOrEqualToTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsLessThanOrEqualToTyped,
    GxB_ISLE
);

// // z = (x|y)
// #[derive(Debug, Clone)]
// pub struct LogicalOr<X, Y, bool>
// where
//     X: ValueType,
//     Y: ValueType,
//     // Z: bool,
// {
//     _value_type_left_input: PhantomData<X>,
//     _value_type_right_input: PhantomData<Y>,
//     _value_type_output: PhantomData<bool>,
// }

// implement_macro_with_2_type_trait_and_output_type_and_typed_graphblas_function_for_all_value_types!(
//     implement_binary_operator,
//     IsLessThan,
//     GrB_LOR,
//     bool
// );

// z = (x|y)
#[derive(Debug, Clone)]
pub struct LogicalOr<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

// implement_binary_operator!(
//     LogicalOr,
//     GrB_LOR,
//     bool,
//     bool,
//     bool,
//     bool,
// );

// z = (x|y)
#[derive(Debug, Clone)]
pub struct LogicalOrTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    LogicalOrTyped,
    GxB_LOR
);

// z = (x&y)
#[derive(Debug, Clone)]
pub struct LogicalAndTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    LogicalAndTyped,
    GxB_LAND
);

// z = (x&y)
#[derive(Debug, Clone)]
pub struct LogicalExclusiveOrTyped<X, Y, Z, T>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
    T: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
    _evaluation_domain: PhantomData<T>,
}

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    LogicalExclusiveOrTyped,
    GxB_LXOR
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_binary_operator() {
        let first = First::<bool, bool, bool, bool>::new();
        let _graphblas_type = first.graphblas_type();

        let plus = Plus::<i8, i8, i8, i8>::new();
        let _graphblas_type = plus.graphblas_type();
    }
}

/*
    // z = x            z = y               z = pow (x,y)
    GrB_FIRST_BOOL,     GrB_SECOND_BOOL,    GxB_POW_BOOL,


    // z = x+y          z = x-y             z = x*y             z = x/y
    GrB_PLUS_BOOL,      GrB_MINUS_BOOL,     GrB_TIMES_BOOL,     GrB_DIV_BOOL,


    // z = y-x          z = y/x             z = 1               z = any(x,y)
    GxB_RMINUS_BOOL,    GxB_RDIV_BOOL,      GxB_PAIR_BOOL,      GxB_ANY_BOOL,


    // The GxB_IS* comparison operators z=f(x,y) return the same type as their
    // inputs.  Each of them compute z = (x OP y), where x, y, and z all have
    // the same type.  The value z is either 1 for true or 0 for false, but it
    // is a value with the same type as x and y.

    // z = (x == y)     z = (x != y)
    GxB_ISEQ_BOOL,      GxB_ISNE_BOOL,


    // z = (x > y)      z = (x < y)         z = (x >= y)     z = (x <= y)
    GxB_ISGT_BOOL,      GxB_ISLT_BOOL,      GxB_ISGE_BOOL,      GxB_ISLE_BOOL,

    // z = min(x,y)     z = max (x,y)
    GrB_MIN_BOOL,       GrB_MAX_BOOL,

    // Binary operators for each of the 11 real types:

    // The operators convert non-boolean types internally to boolean and return
    // a value 1 or 0 in the same type, for true or false.  Each computes z =
    // ((x != 0) OP (y != 0)), where x, y, and z all the same type.  These
    // operators are useful as multiplicative operators when combined with
    // non-boolean monoids of the same type.

    // z = (x || y)     z = (x && y)        z = (x != y)
    GxB_LOR_BOOL,       GxB_LAND_BOOL,      GxB_LXOR_BOOL,

    // Binary operators that operate only on boolean types: LOR, LAND, LXOR,
    // and LXNOR.  The naming convention differs (_BOOL is not appended to the
    // name).  They are the same as GxB_LOR_BOOL, GxB_LAND_BOOL, and
    // GxB_LXOR_BOOL, and GrB_EQ_BOOL, respectively.

    // z = (x || y)     z = (x && y)        z = (x != y)        z = (x == y)
    GrB_LOR,            GrB_LAND,           GrB_LXOR,           GrB_LXNOR,

    // Operators for floating-point reals:

    // z = atan2(x,y)   z = hypot(x,y)      z = fmod(x,y)   z = remainder(x,y)
    GxB_ATAN2_FP32,     GxB_HYPOT_FP32,     GxB_FMOD_FP32,  GxB_REMAINDER_FP32,
    GxB_ATAN2_FP64,     GxB_HYPOT_FP64,     GxB_FMOD_FP64,  GxB_REMAINDER_FP64,

    // z = ldexp(x,y)   z = copysign (x,y)
    GxB_LDEXP_FP32,     GxB_COPYSIGN_FP32,
    GxB_LDEXP_FP64,     GxB_COPYSIGN_FP64,

    // Bitwise operations on signed and unsigned integers: note that
    // bitwise operations on signed integers can lead to different results,
    // depending on your compiler; results are implementation-defined.

    // z = (x | y)      z = (x & y)         z = (x ^ y)        z = ~(x ^ y)
    GrB_BOR_INT8,       GrB_BAND_INT8,      GrB_BXOR_INT8,     GrB_BXNOR_INT8,

    // z = bitget(x,y)  z = bitset(x,y)     z = bitclr(x,y)
    GxB_BGET_INT8,      GxB_BSET_INT8,      GxB_BCLR_INT8,


    // z = bitshift(x,y)
    GxB_BSHIFT_INT8,


    // z = (x == y)     z = (x != y)        z = (x > y)         z = (x < y)
    GrB_EQ_BOOL,        GrB_NE_BOOL,        GrB_GT_BOOL,        GrB_LT_BOOL,


    // z = (x >= y)     z = (x <= y)
    GrB_GE_BOOL,        GrB_LE_BOOL,


    // z = cmplx (x,y)
    GxB_CMPLX_FP32,

GxB_FIRSTI_INT32,   GxB_FIRSTI_INT64,    // z = first_i(A(i,j),y) == i
GxB_FIRSTI1_INT32,  GxB_FIRSTI1_INT64,   // z = first_i1(A(i,j),y) == i+1
GxB_FIRSTJ_INT32,   GxB_FIRSTJ_INT64,    // z = first_j(A(i,j),y) == j
GxB_FIRSTJ1_INT32,  GxB_FIRSTJ1_INT64,   // z = first_j1(A(i,j),y) == j+1
GxB_SECONDI_INT32,  GxB_SECONDI_INT64,   // z = second_i(x,B(i,j)) == i
GxB_SECONDI1_INT32, GxB_SECONDI1_INT64,  // z = second_i1(x,B(i,j)) == i+1
GxB_SECONDJ_INT32,  GxB_SECONDJ_INT64,   // z = second_j(x,B(i,j)) == j
GxB_SECONDJ1_INT32, GxB_SECONDJ1_INT64 ; // z = second_j1(x,B(i,j)) == j+1

*/
