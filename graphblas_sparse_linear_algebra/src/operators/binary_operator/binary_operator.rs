use std::marker::PhantomData;
use std::ptr;

use crate::graphblas_bindings::*;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types,
    implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types,
};
use crate::value_type::ValueType;

pub trait AccumulatorBinaryOperator<T>
where
    T: ValueType,
{
    fn accumulator_graphblas_type(&self) -> GrB_BinaryOp;
}

pub trait BinaryOperator<T>: AccumulatorBinaryOperator<T>
where
    T: ValueType,
{
    fn graphblas_type(&self) -> GrB_BinaryOp;
}

pub trait ReturnsBool {}

macro_rules! implement_binary_operator {
    ($operator_name:ident, $graphblas_operator_trait_name:ident) => {
        pub trait $graphblas_operator_trait_name<T: ValueType> {
            fn graphblas_type() -> GrB_BinaryOp;
        }

        impl<T: ValueType + $graphblas_operator_trait_name<T>> BinaryOperator<T>
            for $operator_name<T>
        {
            fn graphblas_type(&self) -> GrB_BinaryOp {
                T::graphblas_type()
            }
        }

        impl<T: ValueType + $graphblas_operator_trait_name<T>> AccumulatorBinaryOperator<T>
            for $operator_name<T>
        {
            fn accumulator_graphblas_type(&self) -> GrB_BinaryOp {
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

macro_rules! implement_typed_binary_operator {
    ($operator_trait_name:ident,
        $graphblas_operator_name:ident,
        $value_type:ty
    ) => {
        impl $operator_trait_name<$value_type> for $value_type {
            fn graphblas_type() -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }
    };
}

// macro_rules! implement_binary_operator_with_bool_return_type {
//     (
//         $operator_name:ident,
//         $graphblas_operator_name:ident,
//         $evaluation_domain: ty
//     ) => {
//         impl BinaryOperator<$evaluation_domain> for $operator_name<$evaluation_domain> {
//             fn graphblas_type(&self) -> GrB_BinaryOp {
//                 unsafe { $graphblas_operator_name }
//             }
//         }

//         impl AccumulatorBinaryOperator<$evaluation_domain> for $operator_name<$evaluation_domain> {
//             fn accumulator_graphblas_type(&self) -> GrB_BinaryOp {
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

//         impl ReturnsBool for $operator_name<$evaluation_domain> {}
//     };
// }

macro_rules! implement_binary_operator_with_bool_return_type {
    ($operator_name:ident, $graphblas_operator_trait_name:ident) => {
        pub trait $graphblas_operator_trait_name<T: ValueType> {
            fn graphblas_type() -> GrB_BinaryOp;
        }

        impl<T: ValueType + $graphblas_operator_trait_name<T>> BinaryOperator<T>
            for $operator_name<T>
        {
            fn graphblas_type(&self) -> GrB_BinaryOp {
                T::graphblas_type()
            }
        }

        impl<T: ValueType + $graphblas_operator_trait_name<T>> AccumulatorBinaryOperator<T>
            for $operator_name<T>
        {
            fn accumulator_graphblas_type(&self) -> GrB_BinaryOp {
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

        impl<T: ValueType> ReturnsBool for $operator_name<T> {}
    };
}

macro_rules! implement_typed_binary_operator_with_bool_return_type {
    ($operator_trait_name:ident,
        $graphblas_operator_name:ident,
        $value_type:ty
    ) => {
        impl $operator_trait_name<$value_type> for $value_type {
            fn graphblas_type() -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }
    };
}

macro_rules! implement_binary_operator_for_boolean {
    (
        $operator_name:ident,
        $graphblas_operator_name:ident
    ) => {
        impl BinaryOperator<bool> for $operator_name<bool> {
            fn graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl AccumulatorBinaryOperator<bool> for $operator_name<bool> {
            fn accumulator_graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl $operator_name<bool> {
            pub fn new() -> Self {
                Self {
                    _evaluation_domain: PhantomData,
                }
            }
        }

        impl ReturnsBool for $operator_name<bool> {}
    };
}

macro_rules! define_binary_operator {
    ($identifier: ident) => {
        #[derive(Debug, Clone)]
        pub struct $identifier<T>
        where
            T: ValueType,
        {
            _evaluation_domain: PhantomData<T>,
        }
    };
}

define_binary_operator!(Assignment);
impl<T: ValueType> AccumulatorBinaryOperator<T> for Assignment<T> {
    fn accumulator_graphblas_type(&self) -> GrB_BinaryOp {
        ptr::null_mut()
    }
}

// TODO
/// Type will be ignored, type-casting depends on ValueType of the input and output collections
impl<T: ValueType> Assignment<T> {
    pub fn new() -> Self {
        Self {
            _evaluation_domain: PhantomData,
        }
    }
}

// x = first(x,y)
define_binary_operator!(First);
implement_binary_operator!(First, FirstTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    FirstTyped,
    GrB_FIRST
);

// y = second(x,y)
define_binary_operator!(Second);
implement_binary_operator!(Second, SecondTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    SecondTyped,
    GrB_SECOND
);

// z = 1
define_binary_operator!(One);
implement_binary_operator!(One, OneTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    OneTyped,
    GrB_ONEB
);

// z = x^y (z = x.pow(y))
define_binary_operator!(Power);
implement_binary_operator!(Power, PowerTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    PowerTyped,
    GxB_POW
);

// z = x+y
define_binary_operator!(Plus);
implement_binary_operator!(Plus, PlusTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    PlusTyped,
    GrB_PLUS
);

// z = x-y
define_binary_operator!(Minus);
implement_binary_operator!(Minus, MinusTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    MinusTyped,
    GrB_MINUS
);

// z = y-x
define_binary_operator!(ReverseMinus);
implement_binary_operator!(ReverseMinus, ReverseMinusTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    ReverseMinusTyped,
    GxB_RMINUS
);

// z = x*y
define_binary_operator!(Times);
implement_binary_operator!(Times, TimesTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TimesTyped,
    GrB_TIMES
);

// z = x/y
define_binary_operator!(Divide);
implement_binary_operator!(Divide, DivideTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    DivideTyped,
    GrB_DIV
);

// z = x/y
define_binary_operator!(ReverseDivide);
implement_binary_operator!(ReverseDivide, ReverseDivideTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    ReverseDivideTyped,
    GxB_RDIV
);

// z = x==y
#[derive(Debug, Clone)]
pub struct IsEqual<T>
where
    T: ValueType,
{
    _evaluation_domain: PhantomData<T>,
}

implement_binary_operator_with_bool_return_type!(IsEqual, IsEqualTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator_with_bool_return_type,
    IsEqualTyped,
    GrB_EQ
);

// z = x==y
define_binary_operator!(TypedIsEqual);
implement_binary_operator!(TypedIsEqual, TypedIsEqualTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedIsEqualTyped,
    GxB_ISEQ
);

// z = x!=y
#[derive(Debug, Clone)]
pub struct IsNotEqual<T>
where
    T: ValueType,
{
    _evaluation_domain: PhantomData<T>,
}

implement_binary_operator_with_bool_return_type!(IsNotEqual, IsNotEqualTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator_with_bool_return_type,
    IsNotEqualTyped,
    GrB_NE
);

// z = x==y
define_binary_operator!(TypedIsNotEqual);
implement_binary_operator!(TypedIsNotEqual, TypedIsNotEqualTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedIsNotEqualTyped,
    GxB_ISNE
);

// z = any(x,y), selected according to fastest computation speed
define_binary_operator!(Any);
implement_binary_operator!(Any, AnyTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    AnyTyped,
    GxB_ANY
);

// z = min(x,y)
define_binary_operator!(Min);
implement_binary_operator!(Min, MinTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    MinTyped,
    GrB_MIN
);

// z = max(x,y)
define_binary_operator!(Max);
implement_binary_operator!(Max, MaxTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    MaxTyped,
    GrB_MAX
);

// z = (x>y)
#[derive(Debug, Clone)]
pub struct IsGreaterThan<T>
where
    T: ValueType,
{
    _evaluation_domain: PhantomData<T>,
}

implement_binary_operator_with_bool_return_type!(IsGreaterThan, IsGreaterThanTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator_with_bool_return_type,
    IsGreaterThanTyped,
    GrB_GT
);

// z = (x>y)
define_binary_operator!(TypedIsGreaterThan);
implement_binary_operator!(TypedIsGreaterThan, TypedIsGreaterThanTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedIsGreaterThanTyped,
    GxB_ISGT
);

// z = (x>=y)
#[derive(Debug, Clone)]
pub struct IsGreaterThanOrEqualTo<T>
where
    T: ValueType,
{
    _evaluation_domain: PhantomData<T>,
}

implement_binary_operator_with_bool_return_type!(
    IsGreaterThanOrEqualTo,
    IsGreaterThanOrEqualToTyped
);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator_with_bool_return_type,
    IsGreaterThanOrEqualToTyped,
    GrB_GE
);

// z = (x>=y)
define_binary_operator!(TypedIsGreaterThanOrEqualTo);
implement_binary_operator!(
    TypedIsGreaterThanOrEqualTo,
    TypedIsGreaterThanOrEqualToTyped
);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedIsGreaterThanOrEqualToTyped,
    GxB_ISGE
);

// z = (x<y)
#[derive(Debug, Clone)]
pub struct IsLessThan<T>
where
    T: ValueType,
{
    _evaluation_domain: PhantomData<T>,
}

implement_binary_operator_with_bool_return_type!(IsLessThan, IsLessThanTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator_with_bool_return_type,
    IsLessThanTyped,
    GrB_LT
);

// z = (x<y)
define_binary_operator!(TypedIsLessThan);
implement_binary_operator!(TypedIsLessThan, TypedIsLessThanTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedIsLessThanTyped,
    GxB_ISLT
);

// z = (x<=y)
#[derive(Debug, Clone)]
pub struct IsLessThanOrEqualTo<T>
where
    T: ValueType,
{
    _evaluation_domain: PhantomData<T>,
}

implement_binary_operator_with_bool_return_type!(IsLessThanOrEqualTo, IsLessThanOrEqualToTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator_with_bool_return_type,
    IsLessThanOrEqualToTyped,
    GrB_LE
);

// z = (x<=y)
define_binary_operator!(TypedIsLessThanOrEqualTo);
implement_binary_operator!(TypedIsLessThanOrEqualTo, TypedIsLessThanOrEqualToTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedIsLessThanOrEqualToTyped,
    GxB_ISLE
);

// z = (x|y)
define_binary_operator!(LogicalOr);
implement_binary_operator_for_boolean!(LogicalOr, GrB_LOR);

// z = (x|y)
define_binary_operator!(TypedLogicalOr);
implement_binary_operator!(TypedLogicalOr, TypedLogicalOrTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedLogicalOrTyped,
    GxB_LOR
);

// z = (x&y)
define_binary_operator!(LogicalAnd);
implement_binary_operator_for_boolean!(LogicalAnd, GrB_LAND);

// z = (x&y)
define_binary_operator!(TypedLogicalAnd);
implement_binary_operator!(TypedLogicalAnd, TypedLogicalAndTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedLogicalAndTyped,
    GxB_LAND
);

// z = (x|y)
define_binary_operator!(LogicalExclusiveOr);
implement_binary_operator_for_boolean!(LogicalExclusiveOr, GrB_LXOR);

// z = (x&y)
define_binary_operator!(TypedLogicalExclusiveOr);
implement_binary_operator!(TypedLogicalExclusiveOr, TypedLogicalExclusiveOrTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_typed_binary_operator,
    TypedLogicalExclusiveOrTyped,
    GxB_LXOR
);

// z = tan^{-1}(y/x)
define_binary_operator!(FloatingPointFourQuadrantArcTangent);
implement_binary_operator!(
    FloatingPointFourQuadrantArcTangent,
    FloatingPointFourQuadrantArcTangentTyped
);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_binary_operator,
    FloatingPointFourQuadrantArcTangentTyped,
    GxB_ATAN2
);

// z = sqrt(x^2 + y^2)
define_binary_operator!(FloatingPointHypotenuse);
implement_binary_operator!(FloatingPointHypotenuse, FloatingPointHypotenuseTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_binary_operator,
    FloatingPointHypotenuseTyped,
    GxB_HYPOT
);

// z = remainder(x,y)
// Distance to multiple of y closest to x
define_binary_operator!(FloatingPointRemainder);
implement_binary_operator!(FloatingPointRemainder, FloatingPointRemainderTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_binary_operator,
    FloatingPointRemainderTyped,
    GxB_REMAINDER
);

// z = fmod(x,y)
// Distance to last full multiple of y smaller than or equal to x
define_binary_operator!(FloatingPointModulus);
implement_binary_operator!(FloatingPointModulus, FloatingPointModulusTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_binary_operator,
    FloatingPointModulusTyped,
    GxB_FMOD
);

// TODO: test for non-integer y
// z = x*2^y
define_binary_operator!(LDExp);
implement_binary_operator!(LDExp, LDExpTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_binary_operator,
    LDExpTyped,
    GxB_LDEXP
);

// z = copysign(x,y) => z~f(magnitude(x), sign(y))
define_binary_operator!(FloatingPointFromMagnitudeAndSign);
implement_binary_operator!(
    FloatingPointFromMagnitudeAndSign,
    FloatingPointFromMagnitudeAndSignTyped
);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_typed_binary_operator,
    FloatingPointFromMagnitudeAndSignTyped,
    GxB_COPYSIGN
);

// NOTE: bitwise operations on signed integers may behave differently between different compilers

define_binary_operator!(BitWiseLogicalOr);
implement_binary_operator!(BitWiseLogicalOr, BitWiseLogicalOrTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_binary_operator,
    BitWiseLogicalOrTyped,
    GrB_BOR
);

define_binary_operator!(BitWiseLogicalAnd);
implement_binary_operator!(BitWiseLogicalAnd, BitWiseLogicalAndTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_binary_operator,
    BitWiseLogicalAndTyped,
    GrB_BAND
);

define_binary_operator!(BitWiseLogicalExclusiveNotOr);
implement_binary_operator!(
    BitWiseLogicalExclusiveNotOr,
    BitWiseLogicalExclusiveNotOrTyped
);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_binary_operator,
    BitWiseLogicalExclusiveNotOrTyped,
    GrB_BXNOR
);

define_binary_operator!(BitWiseLogicalExclusiveOr);
implement_binary_operator!(BitWiseLogicalExclusiveOr, BitWiseLogicalExclusiveOrTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_binary_operator,
    BitWiseLogicalExclusiveOrTyped,
    GrB_BXOR
);

define_binary_operator!(GetBit);
implement_binary_operator!(GetBit, GetBitTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_binary_operator,
    GetBitTyped,
    GxB_BGET
);

define_binary_operator!(SetBit);
implement_binary_operator!(SetBit, SetBitTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_binary_operator,
    SetBitTyped,
    GxB_BSET
);

define_binary_operator!(ClearBit);
implement_binary_operator!(ClearBit, ClearBitTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_binary_operator,
    ClearBitTyped,
    GxB_BCLR
);

// TODO: consider restricting input to u8. This would improve performance and predictability
define_binary_operator!(ShiftBit);
implement_binary_operator!(ShiftBit, ShiftBitTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_typed_binary_operator,
    ShiftBitTyped,
    GxB_BSHIFT
);

define_binary_operator!(RowIndexFirstArgument);
implement_binary_operator!(RowIndexFirstArgument, RowIndexFirstArgumentTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_typed_binary_operator,
    RowIndexFirstArgumentTyped,
    GxB_FIRSTI
);

define_binary_operator!(ColumnIndexFirstArgument);
implement_binary_operator!(ColumnIndexFirstArgument, ColumnIndexFirstArgumentTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_typed_binary_operator,
    ColumnIndexFirstArgumentTyped,
    GxB_FIRSTJ
);

define_binary_operator!(RowIndexSecondArgument);
implement_binary_operator!(RowIndexSecondArgument, RowIndexSecondArgumentTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_typed_binary_operator,
    RowIndexSecondArgumentTyped,
    GxB_SECONDI
);

define_binary_operator!(ColumnIndexSecondArgument);
implement_binary_operator!(ColumnIndexSecondArgument, ColumnIndexSecondArgumentTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_typed_binary_operator,
    ColumnIndexSecondArgumentTyped,
    GxB_SECONDJ
);

#[cfg(test)]
mod tests {
    use crate::{
        collections::{
            sparse_matrix::{
                operations::{FromMatrixElementList, GetSparseMatrixElementValue},
                MatrixElementList, Size, SparseMatrix,
            },
            sparse_scalar::SparseScalar,
            sparse_vector::{
                operations::{FromVectorElementList, GetVectorElementValue},
                SparseVector, VectorElementList,
            },
            Collection,
        },
        context::{Context, Mode},
        operators::{
            apply::{ApplyBinaryOperatorWithSparseScalar, BinaryOperatorApplier},
            mask::{SelectEntireMatrix, SelectEntireVector},
            options::OperatorOptions,
        },
    };

    use super::*;

    #[test]
    fn new_binary_operator() {
        let first = First::<bool>::new();
        let _graphblas_type = first.graphblas_type();

        let plus = Plus::<i8>::new();
        let _graphblas_type = plus.graphblas_type();
    }

    #[test]
    fn test_divide_by_zero() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list =
            VectorElementList::<u8>::from_element_vector(vec![(1, 1).into(), (2, 0).into()]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_argument = SparseScalar::<u8>::from_value(&context, 0).unwrap();
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &Divide::<u8>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);
        assert_eq!(
            product_vector.get_element_value_or_default(&1).unwrap(),
            u8::MAX
        );
        assert_eq!(product_vector.get_element_value(&2).unwrap(), Some(0));

        let operator = BinaryOperatorApplier::new();

        let mut product_vector = SparseVector::<f32>::new(&context, &vector_length).unwrap();
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &Divide::<u8>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);
        assert_eq!(
            product_vector.get_element_value_or_default(&1).unwrap(),
            u8::MAX as f32
        );
        assert_eq!(
            product_vector.get_element_value_or_default(&2).unwrap(),
            0f32
        );

        let operator = BinaryOperatorApplier::new();
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &Divide::<f32>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();
        println!("{}", product_vector);
        assert_eq!(
            product_vector.get_element_value_or_default(&1).unwrap(),
            f32::INFINITY
        );
        assert!(f32::is_nan(
            product_vector.get_element_value_or_default(&2).unwrap()
        ));
    }

    #[test]
    fn test_ldexp_and_type_casting_for_floating_point_operators() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (0, 1).into(),
            (1, 2).into(),
            (2, 3).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<u8>::from_element_list(
            &context.to_owned(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_argument = SparseScalar::<f32>::from_value(&context, 0.5).unwrap();
        operator
            .apply_with_vector_as_left_argument(
                &vector,
                &LDExp::<f32>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_vector,
                &SelectEntireVector::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_vector);
        assert_eq!(product_vector.get_element_value_or_default(&0).unwrap(), 1);
        assert_eq!(product_vector.get_element_value_or_default(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value_or_default(&2).unwrap(), 3);
    }

    #[test]
    fn test_bitshift() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<f64>::from_element_vector(vec![
            (0, u16::MAX as f64).into(),
            (1, u16::MAX as f64).into(),
            (2, u16::MAX as f64 + 0.5).into(),
        ]);

        let vector_length: usize = 10;
        let vector = SparseVector::<f64>::from_element_list(
            &context.to_owned(),
            &vector_length,
            &element_list,
            &First::<f64>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<f32>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new();

        for i in 0..3 {
            let second_argument =
                SparseScalar::<f32>::from_value(&context, i as f32 + 0.5).unwrap();
            operator
                .apply_with_vector_as_left_argument(
                    &vector,
                    &ShiftBit::<u8>::new(),
                    &second_argument,
                    &Assignment::new(),
                    &mut product_vector,
                    &SelectEntireVector::new(&context),
                    &OperatorOptions::new_default(),
                )
                .unwrap();

            println!("{}", product_vector);
            match i {
                0 => {
                    assert_eq!(
                        product_vector.get_element_value_or_default(&2).unwrap(),
                        255f32
                    );
                }
                1 => {
                    assert_eq!(
                        product_vector.get_element_value_or_default(&2).unwrap(),
                        254f32
                    );
                }
                2 => {
                    assert_eq!(
                        product_vector.get_element_value_or_default(&2).unwrap(),
                        252f32
                    );
                }
                3 => {
                    assert_eq!(
                        product_vector.get_element_value_or_default(&2).unwrap(),
                        248f32
                    );
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_index() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 1, 2).into(),
            (4, 2, 4).into(),
            (5, 2, 5).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.to_owned(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let operator = BinaryOperatorApplier::new();

        let second_argument = SparseScalar::<u8>::from_value(&context, 10).unwrap();
        operator
            .apply_with_matrix_as_left_argument(
                &matrix,
                &RowIndexFirstArgument::<i64>::new(),
                &second_argument,
                &Assignment::new(),
                &mut product_matrix,
                &SelectEntireMatrix::new(&context),
                &OperatorOptions::new_default(),
            )
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.element_value_or_default(&1, &1).unwrap(), 1);
        assert_eq!(product_matrix.element_value_or_default(&2, &1).unwrap(), 2);
        assert_eq!(product_matrix.element_value_or_default(&4, &2).unwrap(), 4);
        assert_eq!(product_matrix.element_value_or_default(&5, &2).unwrap(), 5);
    }
}
