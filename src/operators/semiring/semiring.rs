use crate::operators::binary_operator::{Plus, Times};
use crate::value_type::ValueType;

use crate::bindings_to_graphblas_implementation::*;

pub trait Semiring<Multiplier, Multiplicant, Product>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
{
    fn graphblas_type(&self) -> GrB_Semiring;
}

macro_rules! implement_semiring_operator {
    ($semiring:ident, $addition_operator:ident,$multiplication_operator:ident,$multiplier:ty,$multiplicant:ty,$product:ty,
        $graphblas_operator:ident) => {
        impl Semiring<$multiplier, $multiplicant, $product>
            for $semiring<$multiplier, $multiplicant, $product>
        {
            fn graphblas_type(&self) -> GrB_Semiring {
                unsafe { $graphblas_operator }
            }
        }

        impl $semiring<$multiplier, $multiplicant, $product> {
            pub fn new() -> Self {
                Self {
                    addition_operator: $addition_operator::<$product, $product, $product>::new(),

                    multiplication_operator: $multiplication_operator::<
                        $multiplier,
                        $multiplicant,
                        $product,
                    >::new(),
                }
            }
        }
    };
}

#[derive(Debug, Clone)]
pub struct PlusTimes<Multiplier, Multiplicant, Product>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
{
    // graphblas_type: GrB_Semiring,
    addition_operator: Plus<Product, Product, Product>,
    multiplication_operator: Times<Multiplier, Multiplicant, Product>,
}

implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    u8,
    u8,
    u8,
    GrB_PLUS_TIMES_SEMIRING_UINT8
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    u16,
    u16,
    u16,
    GrB_PLUS_TIMES_SEMIRING_UINT16
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    u32,
    u32,
    u32,
    GrB_PLUS_TIMES_SEMIRING_UINT32
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    u64,
    u64,
    u64,
    GrB_PLUS_TIMES_SEMIRING_UINT64
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    i8,
    i8,
    i8,
    GrB_PLUS_TIMES_SEMIRING_INT8
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    i16,
    i16,
    i16,
    GrB_PLUS_TIMES_SEMIRING_INT16
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    i32,
    i32,
    i32,
    GrB_PLUS_TIMES_SEMIRING_INT32
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    i64,
    i64,
    i64,
    GrB_PLUS_TIMES_SEMIRING_INT64
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    f32,
    f32,
    f32,
    GrB_PLUS_TIMES_SEMIRING_FP32
);
implement_semiring_operator!(
    PlusTimes,
    Plus,
    Times,
    f64,
    f64,
    f64,
    GrB_PLUS_TIMES_SEMIRING_FP64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_semiring() {
        let semiring = PlusTimes::<i8, i8, i8>::new();

        unsafe {
            assert_eq!(semiring.graphblas_type(), GrB_PLUS_TIMES_SEMIRING_INT8);
            assert_ne!(semiring.graphblas_type(), GrB_PLUS_TIMES_SEMIRING_INT16);
        }
    }

    #[test]
    fn test_new_binary_operator() {
        let _plus = Plus::<i8, i8, i8>::new();
    }
}
