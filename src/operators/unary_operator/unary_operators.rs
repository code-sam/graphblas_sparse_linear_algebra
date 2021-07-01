use std::marker::PhantomData;

use crate::bindings_to_graphblas_implementation::*;

use crate::value_type::ValueType;

pub trait UnaryOperator<T>
where
    T: ValueType,
{
    fn graphblas_type(&self) -> GrB_UnaryOp;
}

macro_rules! implement_unary_operator {
    ($operator_name:ident,
        $graphblas_operator_name:ident,
        $value_type:ty
    ) => {
        impl UnaryOperator<$value_type> for $operator_name<$value_type> {
            fn graphblas_type(&self) -> GrB_UnaryOp {
                unsafe { $graphblas_operator_name }
            }
        }

        impl $operator_name<$value_type> {
            pub fn new() -> Self {
                Self {
                    _value_type: PhantomData,
                }
            }
        }
    };
}

///z = x
#[derive(Debug, Clone)]
pub struct Identity<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_unary_operator!(Identity, GrB_IDENTITY_BOOL, bool);
implement_unary_operator!(Identity, GrB_IDENTITY_INT8, i8);
implement_unary_operator!(Identity, GrB_IDENTITY_INT16, i16);
implement_unary_operator!(Identity, GrB_IDENTITY_INT32, i32);
implement_unary_operator!(Identity, GrB_IDENTITY_INT64, i64);
implement_unary_operator!(Identity, GrB_IDENTITY_UINT8, u8);
implement_unary_operator!(Identity, GrB_IDENTITY_UINT16, u16);
implement_unary_operator!(Identity, GrB_IDENTITY_UINT32, u32);
implement_unary_operator!(Identity, GrB_IDENTITY_UINT64, u64);
implement_unary_operator!(Identity, GrB_IDENTITY_FP32, f32);
implement_unary_operator!(Identity, GrB_IDENTITY_FP64, f64);

///z = -x
#[derive(Debug, Clone)]
pub struct AdditiveInverse<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_unary_operator!(AdditiveInverse, GrB_AINV_BOOL, bool);
implement_unary_operator!(AdditiveInverse, GrB_AINV_INT8, i8);
implement_unary_operator!(AdditiveInverse, GrB_AINV_INT16, i16);
implement_unary_operator!(AdditiveInverse, GrB_AINV_INT32, i32);
implement_unary_operator!(AdditiveInverse, GrB_AINV_INT64, i64);
implement_unary_operator!(AdditiveInverse, GrB_AINV_UINT8, u8);
implement_unary_operator!(AdditiveInverse, GrB_AINV_UINT16, u16);
implement_unary_operator!(AdditiveInverse, GrB_AINV_UINT32, u32);
implement_unary_operator!(AdditiveInverse, GrB_AINV_UINT64, u64);
implement_unary_operator!(AdditiveInverse, GrB_AINV_FP32, f32);
implement_unary_operator!(AdditiveInverse, GrB_AINV_FP64, f64);

///z = 1/x
#[derive(Debug, Clone)]
pub struct MultiplicativeInverse<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_unary_operator!(MultiplicativeInverse, GrB_MINV_BOOL, bool);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_INT8, i8);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_INT16, i16);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_INT32, i32);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_INT64, i64);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_UINT8, u8);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_UINT16, u16);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_UINT32, u32);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_UINT64, u64);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_FP32, f32);
implement_unary_operator!(MultiplicativeInverse, GrB_MINV_FP64, f64);

/// z = ! (x != 0)
#[derive(Debug, Clone)]
pub struct LogicalNegation<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_unary_operator!(LogicalNegation, GrB_LNOT, bool);
implement_unary_operator!(LogicalNegation, GxB_LNOT_INT8, i8);
implement_unary_operator!(LogicalNegation, GxB_LNOT_INT16, i16);
implement_unary_operator!(LogicalNegation, GxB_LNOT_INT32, i32);
implement_unary_operator!(LogicalNegation, GxB_LNOT_INT64, i64);
implement_unary_operator!(LogicalNegation, GxB_LNOT_UINT8, u8);
implement_unary_operator!(LogicalNegation, GxB_LNOT_UINT16, u16);
implement_unary_operator!(LogicalNegation, GxB_LNOT_UINT32, u32);
implement_unary_operator!(LogicalNegation, GxB_LNOT_UINT64, u64);
implement_unary_operator!(LogicalNegation, GxB_LNOT_FP32, f32);
implement_unary_operator!(LogicalNegation, GxB_LNOT_FP64, f64);

/// z = 1
/// Only operators in non-zero elements
#[derive(Debug, Clone)]
pub struct One<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_unary_operator!(One, GxB_ONE_BOOL, bool);
implement_unary_operator!(One, GxB_ONE_INT8, i8);
implement_unary_operator!(One, GxB_ONE_INT16, i16);
implement_unary_operator!(One, GxB_ONE_INT32, i32);
implement_unary_operator!(One, GxB_ONE_INT64, i64);
implement_unary_operator!(One, GxB_ONE_UINT8, u8);
implement_unary_operator!(One, GxB_ONE_UINT16, u16);
implement_unary_operator!(One, GxB_ONE_UINT32, u32);
implement_unary_operator!(One, GxB_ONE_UINT64, u64);
implement_unary_operator!(One, GxB_ONE_FP32, f32);
implement_unary_operator!(One, GxB_ONE_FP64, f64);

///z = abs(x)
#[derive(Debug, Clone)]
pub struct AbsoluteValue<T: ValueType> {
    _value_type: PhantomData<T>,
}

implement_unary_operator!(AbsoluteValue, GrB_ABS_BOOL, bool);
implement_unary_operator!(AbsoluteValue, GrB_ABS_INT8, i8);
implement_unary_operator!(AbsoluteValue, GrB_ABS_INT16, i16);
implement_unary_operator!(AbsoluteValue, GrB_ABS_INT32, i32);
implement_unary_operator!(AbsoluteValue, GrB_ABS_INT64, i64);
implement_unary_operator!(AbsoluteValue, GrB_ABS_UINT8, u8);
implement_unary_operator!(AbsoluteValue, GrB_ABS_UINT16, u16);
implement_unary_operator!(AbsoluteValue, GrB_ABS_UINT32, u32);
implement_unary_operator!(AbsoluteValue, GrB_ABS_UINT64, u64);
implement_unary_operator!(AbsoluteValue, GrB_ABS_FP32, f32);
implement_unary_operator!(AbsoluteValue, GrB_ABS_FP64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_binary_operator() {
        let min_monoid = AdditiveInverse::<f32>::new();
        let _graphblas_type = min_monoid.graphblas_type();
    }
}

/*
//------------------------------------------------------------------------------
// built-in unary operators, z = f(x)
//------------------------------------------------------------------------------

// For these functions z=f(x), z and x have the same type.
// The suffix in the name is the type of x and z.
// z = x             z = -x             z = 1/x             z = ! (x != 0)
// identity          additive           multiplicative      logical
//                   inverse            inverse             negation
GrB_IDENTITY_BOOL,   GrB_AINV_BOOL,     GrB_MINV_BOOL,      GxB_LNOT_BOOL,
GrB_IDENTITY_INT8,   GrB_AINV_INT8,     GrB_MINV_INT8,      GxB_LNOT_INT8,
GrB_IDENTITY_INT16,  GrB_AINV_INT16,    GrB_MINV_INT16,     GxB_LNOT_INT16,
GrB_IDENTITY_INT32,  GrB_AINV_INT32,    GrB_MINV_INT32,     GxB_LNOT_INT32,
GrB_IDENTITY_INT64,  GrB_AINV_INT64,    GrB_MINV_INT64,     GxB_LNOT_INT64,
GrB_IDENTITY_UINT8,  GrB_AINV_UINT8,    GrB_MINV_UINT8,     GxB_LNOT_UINT8,
GrB_IDENTITY_UINT16, GrB_AINV_UINT16,   GrB_MINV_UINT16,    GxB_LNOT_UINT16,
GrB_IDENTITY_UINT32, GrB_AINV_UINT32,   GrB_MINV_UINT32,    GxB_LNOT_UINT32,
GrB_IDENTITY_UINT64, GrB_AINV_UINT64,   GrB_MINV_UINT64,    GxB_LNOT_UINT64,
GrB_IDENTITY_FP32,   GrB_AINV_FP32,     GrB_MINV_FP32,      GxB_LNOT_FP32,
GrB_IDENTITY_FP64,   GrB_AINV_FP64,     GrB_MINV_FP64,      GxB_LNOT_FP64,
// complex unary operators:
GxB_IDENTITY_FC32,   GxB_AINV_FC32,     GxB_MINV_FC32,      // no LNOT
GxB_IDENTITY_FC64,   GxB_AINV_FC64,     GxB_MINV_FC64,      // for complex

// z = 1             z = abs(x)         z = bnot(x)         z = signum
// one               absolute value     bitwise negation
GxB_ONE_BOOL,        GrB_ABS_BOOL,
GxB_ONE_INT8,        GrB_ABS_INT8,      GrB_BNOT_INT8,
GxB_ONE_INT16,       GrB_ABS_INT16,     GrB_BNOT_INT16,
GxB_ONE_INT32,       GrB_ABS_INT32,     GrB_BNOT_INT32,
GxB_ONE_INT64,       GrB_ABS_INT64,     GrB_BNOT_INT64,
GxB_ONE_UINT8,       GrB_ABS_UINT8,     GrB_BNOT_UINT8,
GxB_ONE_UINT16,      GrB_ABS_UINT16,    GrB_BNOT_UINT16,
GxB_ONE_UINT32,      GrB_ABS_UINT32,    GrB_BNOT_UINT32,
GxB_ONE_UINT64,      GrB_ABS_UINT64,    GrB_BNOT_UINT64,
GxB_ONE_FP32,        GrB_ABS_FP32,
GxB_ONE_FP64,        GrB_ABS_FP64,
// complex unary operators:
GxB_ONE_FC32,        // for complex types, z = abs(x)
GxB_ONE_FC64,        // is real; listed below.

// Boolean negation, z = !x, where both z and x are boolean.  There is no
// suffix since z and x are only boolean.  This operator is identical to
// GxB_LNOT_BOOL; it just has a different name.
GrB_LNOT ;
*/
