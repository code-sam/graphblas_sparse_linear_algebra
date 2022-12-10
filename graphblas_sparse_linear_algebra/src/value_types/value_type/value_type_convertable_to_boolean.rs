use crate::value_types::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types;

use super::{BuiltInValueType, ValueType};

pub trait AsBoolean<T>: ValueType + BuiltInValueType {}

macro_rules! implement_as_boolean {
    ($value_type: ty) => {
        impl AsBoolean<$value_type> for $value_type {}
    };
}

implement_macro_for_all_value_types!(implement_as_boolean);
