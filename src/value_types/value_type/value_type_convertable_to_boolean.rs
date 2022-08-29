// extern crate proc_macro;
// use proc_macro::TokenStream;

use super::value_type::{BuiltInValueType, ValueType};

pub trait AsBoolean<T>: ValueType + BuiltInValueType<T> {}

// // https://doc.rust-lang.org/reference/procedural-macros.html
// #[proc_macro]
// pub fn supported_types(_item: TokenStream) -> TokenStream {
//     "bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64".parse().unwrap()
// }

macro_rules! implement_as_boolean {
    ($value_type: ty) => {
        impl AsBoolean<$value_type> for $value_type {}
    };
}

// implement_as_boolean!(bool);
// implement_as_boolean!(i8);
// implement_as_boolean!(i16);
// implement_as_boolean!(i32);
// implement_as_boolean!(i64);
// implement_as_boolean!(u8);
// implement_as_boolean!(u16);
// implement_as_boolean!(u32);
// implement_as_boolean!(u64);
// implement_as_boolean!(f32);
// implement_as_boolean!(f64);
// implement_as_boolean!(usize);
// implement_as_boolean!(isize);

// static types: [&str; 3] = ["bool", "i8", "i16"];

macro_rules! generate_types {
    () => {
        bool, i8, i16
    };
}

// macro_rules! generate_implementations {
//     ($($type: ty),*) => {
//         $(
//             implement_as_boolean!($type);
//             // fn $func() {
//             //     println!("I'm a function and my name is {}", stringify!($func));
//             // }
//         )*

//         // let $var = [$(stringify!($func)),*];
//     }
// }

// macro_rules! generate_implementations {
//     ($($function_description: ident),*) => {
//         $(
//             $function_description!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
//             // fn $func() {
//             //     println!("I'm a function and my name is {}", stringify!($func));
//             // }
//         )*

//         // let $var = [$(stringify!($func)),*];
//     }
// }

macro_rules! implement_for_value_types {
    ($id: ident, $($type: ty),*) => {
        $(
            $id!($type);
            // fn $func() {
            //     println!("I'm a function and my name is {}", stringify!($func));
            // }
        )*

        // let $var = [$(stringify!($func)),*];
    }
}

// https://github.com/fabianmurariu/rustgraphblas/blob/d93593b6ea2d50a5469af3903e809bbd22294f3d/src/ops/ffi.rs

implement_for_value_types!(
    implement_as_boolean,
    bool,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f32,
    f64
);
// generate_implementations!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
// generate_implementations!(implement_as_boolean);
