use super::value_type::{BuiltInValueType, ValueType};
use crate::value_types::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types;

pub trait AsBoolean<T>: ValueType + BuiltInValueType {}

macro_rules! implement_as_boolean {
    ($value_type: ty) => {
        impl AsBoolean<$value_type> for $value_type {}
    };
}

implement_macro_for_all_value_types!(implement_as_boolean);
