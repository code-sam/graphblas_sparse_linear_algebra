use super::value_type::{BuiltInValueType, ValueType};

pub trait AsBoolean<T>: ValueType + BuiltInValueType<T> {}

macro_rules! implement_as_boolean {
    ($value_type: ty) => {
        impl AsBoolean<$value_type> for $value_type {}
    };
}

implement_as_boolean!(bool);
implement_as_boolean!(i8);
implement_as_boolean!(i16);
implement_as_boolean!(i32);
implement_as_boolean!(i64);
implement_as_boolean!(u8);
implement_as_boolean!(u16);
implement_as_boolean!(u32);
implement_as_boolean!(u64);
implement_as_boolean!(f32);
implement_as_boolean!(f64);
