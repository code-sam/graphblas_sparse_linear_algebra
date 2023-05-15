use std::marker::PhantomData;

use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean;
use crate::value_type::ValueType;

use crate::bindings_to_graphblas_implementation::*;

pub trait Semiring<EvaluationDomain>
where
    EvaluationDomain: ValueType,
{
    fn graphblas_type(&self) -> GrB_Semiring;
}

macro_rules! define_semiring {
    ($semiring:ident) => {
        #[derive(Debug, Clone)]
        pub struct $semiring<EvaluationDomain>
        where
            EvaluationDomain: ValueType,
        {
            _evaluation_domain: PhantomData<EvaluationDomain>,
        }
    };
}

macro_rules! implement_semiring {
    ($semiring:ident, $graphblas_operator:ident, $evaluation_domain:ty) => {
        impl $semiring<$evaluation_domain> {
            pub fn new() -> Self {
                Self {
                    _evaluation_domain: PhantomData,
                }
            }
        }

        impl Semiring<$evaluation_domain> for $semiring<$evaluation_domain> {
            fn graphblas_type(&self) -> GrB_Semiring {
                unsafe { $graphblas_operator }
            }
        }
    };
}

// MAX

define_semiring!(MaxFirst);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MaxFirst,
    GrB_MAX_FIRST_SEMIRING
);

define_semiring!(MaxSecond);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MaxSecond,
    GrB_MAX_SECOND_SEMIRING
);

// define_semiring!(MaxOne);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxOne,
//     GrB_MAX_ONEB_SEMIRING
// );

define_semiring!(MaxMin);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MaxMin,
    GrB_MAX_MIN_SEMIRING
);

// define_semiring!(MaxMax);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxMax,
//     GrB_MAX_MAX_SEMIRING
// );

define_semiring!(MaxPlus);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MaxPlus,
    GrB_MAX_PLUS_SEMIRING
);

// define_semiring!(MaxMaxus);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxMaxus,
//     GrB_MAX_MINUS_SEMIRING
// );

// define_semiring!(MaxRightMaxus);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxRightMaxus,
//     GrB_MAX_RMINUS_SEMIRING
// );

define_semiring!(MaxTimes);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MaxTimes,
    GrB_MAX_TIMES_SEMIRING
);

// define_semiring!(MaxDivide);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxDivide,
//     GrB_MAX_DIVIDE_SEMIRING
// );

// define_semiring!(MaxRightDivide);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxRightDivide,
//     GrB_MAX_RDIVIDE_SEMIRING
// );

// define_semiring!(MaxIsEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxIsEqualTo,
//     GrB_MAX_ISEQ_SEMIRING
// );

// define_semiring!(MaxIsNotEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxIsNotEqualTo,
//     GrB_MAX_ISNE_SEMIRING
// );

// define_semiring!(MaxIsLessThan);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxIsLessThan,
//     GrB_MAX_ISLT_SEMIRING
// );

// define_semiring!(MaxIsGreaterThan);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxIsGreaterThan,
//     GrB_MAX_ISGT_SEMIRING
// );

// define_semiring!(MaxIsLessThanOrEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxIsLessThanOrEqualTo,
//     GrB_MAX_ISLE_SEMIRING
// );

// define_semiring!(MaxIsGreaterThanOrEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxIsGreaterThanOrEqualTo,
//     GrB_MAX_ISGE_SEMIRING
// );

// define_semiring!(MaxLogicalOr);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxLogicalOr,
//     GrB_MAX_LOR_SEMIRING
// );

// define_semiring!(MaxLogicalAnd);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxLogicalAnd,
//     GrB_MAX_LAND_SEMIRING
// );

// define_semiring!(MaxLogicalExclusiveOr);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MaxLogicalExclusiveOr,
//     GrB_MAX_LXOR_SEMIRING
// );

// MIN

define_semiring!(MinFirst);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MinFirst,
    GrB_MIN_FIRST_SEMIRING
);

define_semiring!(MinSecond);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MinSecond,
    GrB_MIN_SECOND_SEMIRING
);

// define_semiring!(MinOne);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinOne,
//     GrB_MIN_ONEB_SEMIRING
// );

// define_semiring!(MinMIN);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinSecond,
//     GrB_MIN_MIN_SEMIRING
// );

define_semiring!(MinMax);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MinMax,
    GrB_MIN_MAX_SEMIRING
);

define_semiring!(MinPlus);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MinPlus,
    GrB_MIN_PLUS_SEMIRING
);

// define_semiring!(MinMinus);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinMinus,
//     GrB_MIN_MINUS_SEMIRING
// );

// define_semiring!(MinRightMinus);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinRightMinus,
//     GrB_MIN_RMINUS_SEMIRING
// );

define_semiring!(MinTimes);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    MinTimes,
    GrB_MIN_TIMES_SEMIRING
);

// define_semiring!(MinDivide);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinDivide,
//     GrB_MIN_DIVIDE_SEMIRING
// );

// define_semiring!(MinRightDivide);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinRightDivide,
//     GrB_MIN_RDIVIDE_SEMIRING
// );

// define_semiring!(MinIsEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinIsEqualTo,
//     GrB_MIN_ISEQ_SEMIRING
// );

// define_semiring!(MinIsNotEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinIsNotEqualTo,
//     GrB_MIN_ISNE_SEMIRING
// );

// define_semiring!(MinIsLessThan);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinIsLessThan,
//     GrB_MIN_ISLT_SEMIRING
// );

// define_semiring!(MinIsGreaterThan);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinIsGreaterThan,
//     GrB_MIN_ISGT_SEMIRING
// );

// define_semiring!(MinIsLessThanOrEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinIsLessThanOrEqualTo,
//     GrB_MIN_ISLE_SEMIRING
// );

// define_semiring!(MinIsGreaterThanOrEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinIsGreaterThanOrEqualTo,
//     GrB_MIN_ISGE_SEMIRING
// );

// define_semiring!(MinLogicalOr);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinLogicalOr,
//     GrB_MIN_LOR_SEMIRING
// );

// define_semiring!(MinLogicalAnd);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinLogicalAnd,
//     GrB_MIN_LAND_SEMIRING
// );

// define_semiring!(MinLogicalExclusiveOr);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     MinLogicalExclusiveOr,
//     GrB_MIN_LXOR_SEMIRING
// );

// PLUS

// define_semiring!(PlusFirst);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusFirst,
//     GrB_PLUS_FIRST_SEMIRING
// );

// define_semiring!(PlusSecond);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusSecond,
//     GrB_PLUS_SECOND_SEMIRING
// );

// define_semiring!(PlusOne);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusOne,
//     GrB_PLUS_ONEB_SEMIRING
// );

// define_semiring!(PlusPLUS);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusSecond,
//     GrB_PLUS_PLUS_SEMIRING
// );

// define_semiring!(PlusMax);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusMax,
//     GrB_PLUS_MAX_SEMIRING
// );

// define_semiring!(PlusPlus);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusPlus,
//     GrB_PLUS_PLUS_SEMIRING
// );

// define_semiring!(PlusPlus);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusPlus,
//     GrB_PLUS_PLUS_SEMIRING
// );

// define_semiring!(PlusRightPlusus);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusRightPlusus,
//     GrB_PLUS_RPLUSUS_SEMIRING
// );

define_semiring!(PlusTimes);
implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
    implement_semiring,
    PlusTimes,
    GrB_PLUS_TIMES_SEMIRING
);

// define_semiring!(PlusDivide);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusDivide,
//     GrB_PLUS_DIVIDE_SEMIRING
// );

// define_semiring!(PlusRightDivide);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusRightDivide,
//     GrB_PLUS_RDIVIDE_SEMIRING
// );

// define_semiring!(PlusIsEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusIsEqualTo,
//     GrB_PLUS_ISEQ_SEMIRING
// );

// define_semiring!(PlusIsNotEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusIsNotEqualTo,
//     GrB_PLUS_ISNE_SEMIRING
// );

// define_semiring!(PlusIsLessThan);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusIsLessThan,
//     GrB_PLUS_ISLT_SEMIRING
// );

// define_semiring!(PlusIsGreaterThan);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusIsGreaterThan,
//     GrB_PLUS_ISGT_SEMIRING
// );

// define_semiring!(PlusIsLessThanOrEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusIsLessThanOrEqualTo,
//     GrB_PLUS_ISLE_SEMIRING
// );

// define_semiring!(PlusIsGreaterThanOrEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusIsGreaterThanOrEqualTo,
//     GrB_PLUS_ISGE_SEMIRING
// );

// define_semiring!(PlusLogicalOr);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusLogicalOr,
//     GrB_PLUS_LOR_SEMIRING
// );

// define_semiring!(PlusLogicalAnd);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusLogicalAnd,
//     GrB_PLUS_LAND_SEMIRING
// );

// define_semiring!(PlusLogicalExclusiveOr);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     PlusLogicalExclusiveOr,
//     GrB_MIN_LXOR_SEMIRING
// );

// TIMES

// define_semiring!(TimesFirst);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesFirst,
//     GrB_TIMES_FIRST_SEMIRING
// );

// define_semiring!(TimesSecond);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesSecond,
//     GrB_TIMES_SECOND_SEMIRING
// );

// define_semiring!(TimesOne);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesOne,
//     GrB_TIMES_ONEB_SEMIRING
// );

// define_semiring!(TimesTIMES);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesSecond,
//     GrB_TIMES_TIMES_SEMIRING
// );

// define_semiring!(TimesMax);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesMax,
//     GrB_TIMES_MAX_SEMIRING
// );

// define_semiring!(TimesTimes);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesTimes,
//     GrB_TIMES_TIMES_SEMIRING
// );

// define_semiring!(TimesTimes);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesTimes,
//     GrB_TIMES_TIMES_SEMIRING
// );

// define_semiring!(TimesRightTimesus);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesRightTimesus,
//     GrB_TIMES_RTIMESUS_SEMIRING
// );

// define_semiring!(TimesTimes);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesTimes,
//     GrB_TIMES_TIMES_SEMIRING
// );

// define_semiring!(TimesDivide);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesDivide,
//     GrB_TIMES_DIVIDE_SEMIRING
// );

// define_semiring!(TimesRightDivide);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesRightDivide,
//     GrB_TIMES_RDIVIDE_SEMIRING
// );

// define_semiring!(TimesIsEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesIsEqualTo,
//     GrB_TIMES_ISEQ_SEMIRING
// );

// define_semiring!(TimesIsNotEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesIsNotEqualTo,
//     GrB_TIMES_ISNE_SEMIRING
// );

// define_semiring!(TimesIsLessThan);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesIsLessThan,
//     GrB_TIMES_ISLT_SEMIRING
// );

// define_semiring!(TimesIsGreaterThan);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesIsGreaterThan,
//     GrB_TIMES_ISGT_SEMIRING
// );

// define_semiring!(TimesIsLessThanOrEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesIsLessThanOrEqualTo,
//     GrB_TIMES_ISLE_SEMIRING
// );

// define_semiring!(TimesIsGreaterThanOrEqualTo);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesIsGreaterThanOrEqualTo,
//     GrB_TIMES_ISGE_SEMIRING
// );

// define_semiring!(TimesLogicalOr);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesLogicalOr,
//     GrB_TIMES_LOR_SEMIRING
// );

// define_semiring!(TimesLogicalAnd);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesLogicalAnd,
//     GrB_PLUS_LAND_SEMIRING
// );

// define_semiring!(TimesLogicalExclusiveOr);
// implement_macro_with_1_type_trait_and_typed_graphblas_function_for_all_value_types_without_boolean!(
//     implement_semiring,
//     TimesLogicalExclusiveOr,
//     GrB_PLUS_LXOR_SEMIRING
// );

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_semiring() {
        let semiring = PlusTimes::<i8>::new();

        unsafe {
            assert_eq!(semiring.graphblas_type(), GrB_PLUS_TIMES_SEMIRING_INT8);
            assert_ne!(semiring.graphblas_type(), GrB_PLUS_TIMES_SEMIRING_INT16);
        }
    }
}
