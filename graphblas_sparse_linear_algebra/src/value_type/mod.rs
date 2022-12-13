mod type_conversion;
// mod custom_value_type;
mod value_type;
mod value_type_convertable_to_boolean;

pub(crate) mod utilities_to_implement_traits_for_all_value_types;

pub(crate) use type_conversion::{ConvertScalar, ConvertVector};
// pub(crate) use custom_value_type::RegisteredCustomValueType;
// pub use built_in_value_type::BuiltInValueType; // private because it exposes to_graphblas_type()
pub use value_type::ValueType;
pub use value_type_convertable_to_boolean::AsBoolean;
