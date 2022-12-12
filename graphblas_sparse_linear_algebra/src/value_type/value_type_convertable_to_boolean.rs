use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types;

use super::ValueType;

pub trait AsBoolean: ValueType {}

macro_rules! implement_as_boolean {
    ($value_type: ty) => {
        impl AsBoolean for $value_type {}
    };
}

implement_macro_for_all_value_types!(implement_as_boolean);
