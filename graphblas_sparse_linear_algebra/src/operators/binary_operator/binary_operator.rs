use std::marker::PhantomData;
use std::ptr;

use crate::bindings_to_graphblas_implementation::*;
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
    (
        $operator_name:ident,
        $graphblas_operator_name:ident,
        $evaluation_domain: ty
    ) => {
        impl BinaryOperator<$evaluation_domain> for $operator_name<$evaluation_domain> {
            fn graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl AccumulatorBinaryOperator<$evaluation_domain> for $operator_name<$evaluation_domain> {
            fn accumulator_graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl $operator_name<$evaluation_domain> {
            pub fn new() -> Self {
                Self {
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
        impl BinaryOperator<$evaluation_domain> for $operator_name<$evaluation_domain> {
            fn graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl AccumulatorBinaryOperator<$evaluation_domain> for $operator_name<$evaluation_domain> {
            fn accumulator_graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl $operator_name<$evaluation_domain> {
            pub fn new() -> Self {
                Self {
                    _evaluation_domain: PhantomData,
                }
            }
        }

        impl ReturnsBool for $operator_name<$evaluation_domain> {}
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

impl<T: ValueType> Assignment<T> {
    pub fn new() -> Self {
        Self {
            _evaluation_domain: PhantomData,
        }
    }
}

// x = first(x,y)
define_binary_operator!(First);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    First,
    GrB_FIRST
);

// y = second(x,y)
define_binary_operator!(Second);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Second,
    GrB_SECOND
);

// z = 1
define_binary_operator!(One);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    One,
    GrB_ONEB
);

// z = x^y (z = x.pow(y))
define_binary_operator!(Power);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Power,
    GxB_POW
);

// z = x+y
define_binary_operator!(Plus);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Plus,
    GrB_PLUS
);

// z = x-y
define_binary_operator!(Minus);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Minus,
    GrB_MINUS
);

// z = y-x
define_binary_operator!(ReverseMinus);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    ReverseMinus,
    GxB_RMINUS
);

// z = x*y
define_binary_operator!(Times);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Times,
    GrB_TIMES
);

// z = x/y
define_binary_operator!(Divide);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Divide,
    GrB_DIV
);

// z = x/y
define_binary_operator!(ReverseDivide);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    ReverseDivide,
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

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsEqual,
    GrB_EQ
);

// z = x==y
define_binary_operator!(IsEqualTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsEqualTyped,
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

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsNotEqual,
    GrB_NE
);

// z = x==y
define_binary_operator!(IsNotEqualTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsNotEqualTyped,
    GxB_ISNE
);

// z = any(x,y), selected according to fastest computation speed
define_binary_operator!(Any);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Any,
    GxB_ANY
);

// z = min(x,y)
define_binary_operator!(Min);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Min,
    GrB_MIN
);

// z = max(x,y)
define_binary_operator!(Max);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    Max,
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

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsGreaterThan,
    GrB_GT
);

// z = (x>y)
define_binary_operator!(IsGreaterThanTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsGreaterThanTyped,
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

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsGreaterThanOrEqualTo,
    GrB_GE
);

// z = (x>=y)
define_binary_operator!(IsGreaterThanOrEqualToTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsGreaterThanOrEqualToTyped,
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

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsLessThan,
    GrB_LT
);

// z = (x<y)
define_binary_operator!(IsLessThanTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsLessThanTyped,
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

implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator_with_bool_return_type,
    IsLessThanOrEqualTo,
    GrB_LE
);

// z = (x<=y)
define_binary_operator!(IsLessThanOrEqualToTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    IsLessThanOrEqualToTyped,
    GxB_ISLE
);

// z = (x|y)
define_binary_operator!(LogicalOr);
implement_binary_operator_for_boolean!(LogicalOr, GrB_LOR);

// z = (x|y)
define_binary_operator!(LogicalOrTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    LogicalOrTyped,
    GxB_LOR
);

// z = (x&y)
define_binary_operator!(LogicalAnd);
implement_binary_operator_for_boolean!(LogicalAnd, GrB_LAND);

// z = (x&y)
define_binary_operator!(LogicalAndTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    LogicalAndTyped,
    GxB_LAND
);

// z = (x|y)
define_binary_operator!(LogicalExclusiveOr);
implement_binary_operator_for_boolean!(LogicalExclusiveOr, GrB_LXOR);

// z = (x&y)
define_binary_operator!(LogicalExclusiveOrTyped);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types!(
    implement_binary_operator,
    LogicalExclusiveOrTyped,
    GxB_LXOR
);

// z = tan^{-1}(y/x)
define_binary_operator!(FloatingPointFourQuadrantArcTangent);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_binary_operator,
    FloatingPointFourQuadrantArcTangent,
    GxB_ATAN2
);

// z = sqrt(x^2 + y^2)
define_binary_operator!(FloatingPointHypotenuse);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_binary_operator,
    FloatingPointHypotenuse,
    GxB_HYPOT
);

// z = remainder(x,y)
// Distance to multiple of y closest to x
define_binary_operator!(FloatingPointRemainder);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_binary_operator,
    FloatingPointRemainder,
    GxB_REMAINDER
);

// z = fmod(x,y)
// Distance to last full multiple of y smaller than or equal to x
define_binary_operator!(FloatingPointModulus);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_binary_operator,
    FloatingPointModulus,
    GxB_FMOD
);

// TODO: test for non-integer y
// z = x*2^y
define_binary_operator!(LDExp);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_binary_operator,
    LDExp,
    GxB_LDEXP
);

// z = copysign(x,y) => z~f(magnitude(x), sign(y))
define_binary_operator!(FloatingPointFromMagnitudeAndSign);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_floating_point_types!(
    implement_binary_operator,
    FloatingPointFromMagnitudeAndSign,
    GxB_COPYSIGN
);

// NOTE: bitwise operations on signed integers may behave differently between different compilers

define_binary_operator!(BitWiseLogicalOr);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_binary_operator,
    BitWiseLogicalOr,
    GrB_BOR
);

define_binary_operator!(BitWiseLogicalAnd);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_binary_operator,
    BitWiseLogicalAnd,
    GrB_BAND
);

define_binary_operator!(BitWiseLogicalExclusiveNotOr);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_binary_operator,
    BitWiseLogicalExclusiveNotOr,
    GrB_BXNOR
);

define_binary_operator!(BitWiseLogicalExclusiveOr);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_binary_operator,
    BitWiseLogicalExclusiveOr,
    GrB_BXOR
);

define_binary_operator!(GetBit);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_binary_operator,
    GetBit,
    GxB_BGET
);

define_binary_operator!(SetBit);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_binary_operator,
    SetBit,
    GxB_BSET
);

define_binary_operator!(ClearBit);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_binary_operator,
    ClearBit,
    GxB_BCLR
);

// TODO: consider restricting input to u8. This would improve performance and predictability
define_binary_operator!(ShiftBit);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_integer_value_types!(
    implement_binary_operator,
    ShiftBit,
    GxB_BSHIFT
);

define_binary_operator!(RowIndexFirstArgument);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_binary_operator,
    RowIndexFirstArgument,
    GxB_FIRSTI
);

define_binary_operator!(ColumnIndexFirstArgument);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_binary_operator,
    ColumnIndexFirstArgument,
    GxB_FIRSTJ
);

define_binary_operator!(RowIndexSecondArgument);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_binary_operator,
    RowIndexSecondArgument,
    GxB_SECONDI
);

define_binary_operator!(ColumnIndexSecondArgument);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_graphblas_index_integer_value_types!(
    implement_binary_operator,
    ColumnIndexSecondArgument,
    GxB_SECONDJ
);

#[cfg(test)]
mod tests {
    use crate::{
        collections::{
            sparse_matrix::{
                FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size, SparseMatrix,
            },
            sparse_scalar::SparseScalar,
            sparse_vector::{
                FromVectorElementList, GetVectorElementValue, SparseVector, VectorElementList,
            },
            Collection,
        },
        context::{Context, Mode},
        operators::{
            apply::{ApplyBinaryOperatorWithSparseScalar, BinaryOperatorApplier},
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
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new(
            &Divide::<u8>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        let second_agrument = SparseScalar::<u8>::from_value(&context, 0).unwrap();
        operator
            .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
            .unwrap();

        println!("{}", product_vector);
        assert_eq!(
            product_vector.get_element_value_or_default(&1).unwrap(),
            u8::MAX
        );
        assert_eq!(product_vector.get_element_value(&2).unwrap(), Some(0));

        let operator = BinaryOperatorApplier::new(
            &Divide::<u8>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        let mut product_vector = SparseVector::<f32>::new(&context, &vector_length).unwrap();
        operator
            .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
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

        let operator = BinaryOperatorApplier::new(
            &Divide::<f32>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );
        operator
            .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
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
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<u8>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new(
            &LDExp::<f32>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        let second_agrument = SparseScalar::<f32>::from_value(&context, 0.5).unwrap();
        operator
            .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
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
            &context.clone(),
            &vector_length,
            &element_list,
            &First::<f64>::new(),
        )
        .unwrap();

        let mut product_vector = SparseVector::<f32>::new(&context, &vector_length).unwrap();

        let operator = BinaryOperatorApplier::new(
            &ShiftBit::<u8>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        for i in 0..3 {
            let second_agrument =
                SparseScalar::<f32>::from_value(&context, i as f32 + 0.5).unwrap();
            operator
                .apply_with_vector_as_first_argument(&vector, &second_agrument, &mut product_vector)
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
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let operator = BinaryOperatorApplier::new(
            &RowIndexFirstArgument::<i64>::new(),
            &OperatorOptions::new_default(),
            &Assignment::new(),
        );

        let second_agrument = SparseScalar::<u8>::from_value(&context, 10).unwrap();
        operator
            .apply_with_matrix_as_first_argument(&matrix, &second_agrument, &mut product_matrix)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(1, 1).into())
                .unwrap(),
            1
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(2, 1).into())
                .unwrap(),
            2
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(4, 2).into())
                .unwrap(),
            4
        );
        assert_eq!(
            product_matrix
                .get_element_value_or_default(&(5, 2).into())
                .unwrap(),
            5
        );
    }
}
