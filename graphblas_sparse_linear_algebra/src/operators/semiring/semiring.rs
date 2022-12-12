use crate::operators::binary_operator::{Plus, Times};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_semiring_for_all_value_types;
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
    ($semiring:ident, $addition_operator:ident, $multiplication_operator:ident, $multiplier:ty, $multiplicant:ty, $product:ty,
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

implement_semiring_for_all_value_types!(
    implement_semiring_operator,
    PlusTimes,
    Plus,
    Times,
    GrB_PLUS_TIMES_SEMIRING
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
