use std::marker::PhantomData;

use crate::bindings_to_graphblas_implementation::*;
use crate::value_types::value_type::ValueType;

pub trait BinaryOperator<X, Y, Z>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
{
    fn graphblas_type(&self) -> GrB_BinaryOp;
}

macro_rules! implement_binary_operator {
    ($operator_name:ident,
        $graphblas_operator_name:ident,
        $value_type_left_input:ty,
        $value_type_right_input:ty,
        $value_type_output:ty
    ) => {
        impl BinaryOperator<$value_type_left_input, $value_type_right_input, $value_type_output>
            for $operator_name<$value_type_left_input, $value_type_right_input, $value_type_output>
        {
            fn graphblas_type(&self) -> GrB_BinaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl $operator_name<$value_type_left_input, $value_type_right_input, $value_type_output> {
            pub fn new() -> Self {
                Self {
                    _value_type_left_input: PhantomData,
                    _value_type_right_input: PhantomData,
                    _value_type_output: PhantomData,
                }
            }
        }
    };
}

// x = first(x,y)
#[derive(Debug, Clone)]
pub struct First<X, Y, Z>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
}

implement_binary_operator!(First, GrB_FIRST_BOOL, bool, bool, bool);
implement_binary_operator!(First, GrB_FIRST_INT8, i8, i8, i8);
implement_binary_operator!(First, GrB_FIRST_INT16, i16, i16, i16);
implement_binary_operator!(First, GrB_FIRST_INT32, i32, i32, i32);
implement_binary_operator!(First, GrB_FIRST_INT64, i64, i64, i64);
implement_binary_operator!(First, GrB_FIRST_UINT8, u8, u8, u8);
implement_binary_operator!(First, GrB_FIRST_UINT16, u16, u16, u16);
implement_binary_operator!(First, GrB_FIRST_UINT32, u32, u32, u32);
implement_binary_operator!(First, GrB_FIRST_UINT64, u64, u64, u64);
implement_binary_operator!(First, GrB_FIRST_FP32, f32, f32, f32);
implement_binary_operator!(First, GrB_FIRST_FP64, f64, f64, f64);

// y = second(x,y)
#[derive(Debug, Clone)]
pub struct Second<X, Y, Z>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
}

implement_binary_operator!(Second, GrB_SECOND_BOOL, bool, bool, bool);
implement_binary_operator!(Second, GrB_SECOND_INT8, i8, i8, i8);
implement_binary_operator!(Second, GrB_SECOND_INT16, i16, i16, i16);
implement_binary_operator!(Second, GrB_SECOND_INT32, i32, i32, i32);
implement_binary_operator!(Second, GrB_SECOND_INT64, i64, i64, i64);
implement_binary_operator!(Second, GrB_SECOND_UINT8, u8, u8, u8);
implement_binary_operator!(Second, GrB_SECOND_UINT16, u16, u16, u16);
implement_binary_operator!(Second, GrB_SECOND_UINT32, u32, u32, u32);
implement_binary_operator!(Second, GrB_SECOND_UINT64, u64, u64, u64);
implement_binary_operator!(Second, GrB_SECOND_FP32, f32, f32, f32);
implement_binary_operator!(Second, GrB_SECOND_FP64, f64, f64, f64);

// z = x^y (z = x.pow(y))
#[derive(Debug, Clone)]
pub struct Power<X, Y, Z>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
}

implement_binary_operator!(Power, GxB_POW_BOOL, bool, bool, bool);
implement_binary_operator!(Power, GxB_POW_INT8, i8, i8, i8);
implement_binary_operator!(Power, GxB_POW_INT16, i16, i16, i16);
implement_binary_operator!(Power, GxB_POW_INT32, i32, i32, i32);
implement_binary_operator!(Power, GxB_POW_INT64, i64, i64, i64);
implement_binary_operator!(Power, GxB_POW_UINT8, u8, u8, u8);
implement_binary_operator!(Power, GxB_POW_UINT16, u16, u16, u16);
implement_binary_operator!(Power, GxB_POW_UINT32, u32, u32, u32);
implement_binary_operator!(Power, GxB_POW_UINT64, u64, u64, u64);
implement_binary_operator!(Power, GxB_POW_FP32, f32, f32, f32);
implement_binary_operator!(Power, GxB_POW_FP64, f64, f64, f64);

// z = x+y
#[derive(Debug, Clone)]
pub struct Plus<X, Y, Z>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
}

implement_binary_operator!(Plus, GrB_PLUS_BOOL, bool, bool, bool);
implement_binary_operator!(Plus, GrB_PLUS_INT8, i8, i8, i8);
implement_binary_operator!(Plus, GrB_PLUS_INT16, i16, i16, i16);
implement_binary_operator!(Plus, GrB_PLUS_INT32, i32, i32, i32);
implement_binary_operator!(Plus, GrB_PLUS_INT64, i64, i64, i64);
implement_binary_operator!(Plus, GrB_PLUS_UINT8, u8, u8, u8);
implement_binary_operator!(Plus, GrB_PLUS_UINT16, u16, u16, u16);
implement_binary_operator!(Plus, GrB_PLUS_UINT32, u32, u32, u32);
implement_binary_operator!(Plus, GrB_PLUS_UINT64, u64, u64, u64);
implement_binary_operator!(Plus, GrB_PLUS_FP32, f32, f32, f32);
implement_binary_operator!(Plus, GrB_PLUS_FP64, f64, f64, f64);

// z = x-y
#[derive(Debug, Clone)]
pub struct Minus<X, Y, Z>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
}

implement_binary_operator!(Minus, GrB_MINUS_BOOL, bool, bool, bool);
implement_binary_operator!(Minus, GrB_MINUS_INT8, i8, i8, i8);
implement_binary_operator!(Minus, GrB_MINUS_INT16, i16, i16, i16);
implement_binary_operator!(Minus, GrB_MINUS_INT32, i32, i32, i32);
implement_binary_operator!(Minus, GrB_MINUS_INT64, i64, i64, i64);
implement_binary_operator!(Minus, GrB_MINUS_UINT8, u8, u8, u8);
implement_binary_operator!(Minus, GrB_MINUS_UINT16, u16, u16, u16);
implement_binary_operator!(Minus, GrB_MINUS_UINT32, u32, u32, u32);
implement_binary_operator!(Minus, GrB_MINUS_UINT64, u64, u64, u64);
implement_binary_operator!(Minus, GrB_MINUS_FP32, f32, f32, f32);
implement_binary_operator!(Minus, GrB_MINUS_FP64, f64, f64, f64);

// z = x*y
#[derive(Debug, Clone)]
pub struct Times<X, Y, Z>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
}

implement_binary_operator!(Times, GrB_TIMES_BOOL, bool, bool, bool);
implement_binary_operator!(Times, GrB_TIMES_INT8, i8, i8, i8);
implement_binary_operator!(Times, GrB_TIMES_INT16, i16, i16, i16);
implement_binary_operator!(Times, GrB_TIMES_INT32, i32, i32, i32);
implement_binary_operator!(Times, GrB_TIMES_INT64, i64, i64, i64);
implement_binary_operator!(Times, GrB_TIMES_UINT8, u8, u8, u8);
implement_binary_operator!(Times, GrB_TIMES_UINT16, u16, u16, u16);
implement_binary_operator!(Times, GrB_TIMES_UINT32, u32, u32, u32);
implement_binary_operator!(Times, GrB_TIMES_UINT64, u64, u64, u64);
implement_binary_operator!(Times, GrB_TIMES_FP32, f32, f32, f32);
implement_binary_operator!(Times, GrB_TIMES_FP64, f64, f64, f64);

// z = x*y
#[derive(Debug, Clone)]
pub struct Divide<X, Y, Z>
where
    X: ValueType,
    Y: ValueType,
    Z: ValueType,
{
    _value_type_left_input: PhantomData<X>,
    _value_type_right_input: PhantomData<Y>,
    _value_type_output: PhantomData<Z>,
}

implement_binary_operator!(Divide, GrB_DIV_BOOL, bool, bool, bool);
implement_binary_operator!(Divide, GrB_DIV_INT8, i8, i8, i8);
implement_binary_operator!(Divide, GrB_DIV_INT16, i16, i16, i16);
implement_binary_operator!(Divide, GrB_DIV_INT32, i32, i32, i32);
implement_binary_operator!(Divide, GrB_DIV_INT64, i64, i64, i64);
implement_binary_operator!(Divide, GrB_DIV_UINT8, u8, u8, u8);
implement_binary_operator!(Divide, GrB_DIV_UINT16, u16, u16, u16);
implement_binary_operator!(Divide, GrB_DIV_UINT32, u32, u32, u32);
implement_binary_operator!(Divide, GrB_DIV_UINT64, u64, u64, u64);
implement_binary_operator!(Divide, GrB_DIV_FP32, f32, f32, f32);
implement_binary_operator!(Divide, GrB_DIV_FP64, f64, f64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_binary_operator() {
        let first = First::<bool, bool, bool>::new();
        let _graphblas_type = first.graphblas_type();

        let plus = Plus::<i8, i8, i8>::new();
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
