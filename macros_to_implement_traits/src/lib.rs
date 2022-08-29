extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, DeriveInput};

// use crate::bindings_to_graphblas_implementation::{
//     GrB_BOOL, GrB_FP32, GrB_FP64, GrB_INT16, GrB_INT32, GrB_INT64, GrB_INT8, GrB_Index, GrB_Type,
//     GrB_Type_free, GrB_UINT16, GrB_UINT32, GrB_UINT64, GrB_UINT8, GxB_Type_size,
// };

// #[proc_macro]
// pub fn supported_types(_item: TokenStream) -> TokenStream {
//     "bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, isize, usize".parse().unwrap()
// }

#[proc_macro]
pub fn graphblas_built_in_type_for_usize(_item: TokenStream) -> TokenStream {
    match usize::BITS {
        8 => "GrB_UINT8".parse().unwrap(),
        16 => "GrB_UINT16".parse().unwrap(),
        32 => "GrB_UINT32".parse().unwrap(),
        64 => "GrB_UINT64".parse().unwrap(),
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    }
}

// #[proc_macro]
// pub fn graphblas_built_in_type_for_isize(_item: TokenStream) -> TokenStream {
//     match isize::BITS {
//         8 => "GrB_INT8".parse::<GrB_Type>().unwrap(),
//         16 => "GrB_INT8".parse::<GrB_Type>().unwrap(),
//         32 => "GrB_INT8".parse::<GrB_Type>().unwrap(),
//         64 => "GrB_INT8".parse::<GrB_Type>().unwrap(),
//         _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
//     }
// }

#[proc_macro]
pub fn implement_macro_for_isize(input: TokenStream) -> TokenStream {
    let macro_identifier = syn::parse_macro_input!(input as syn::Ident);

    let expanded = match isize::BITS {
        8 => {
            quote! {
                #macro_identifier!(isize, GrB_INT8);
            }
        }
        16 => {
            quote! {
                #macro_identifier!(isize, GrB_INT16);
            }
        }
        32 => {
            quote! {
                #macro_identifier!(isize, GrB_INT32);
            }
        }
        64 => {
            quote! {
                #macro_identifier!(isize, GrB_INT64);
            }
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };

    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_usize(input: TokenStream) -> TokenStream {
    let macro_identifier = syn::parse_macro_input!(input as syn::Ident);

    let expanded = match usize::BITS {
        8 => {
            quote! {
                #macro_identifier!(usize, GrB_UINT8);
            }
        }
        16 => {
            quote! {
                #macro_identifier!(usize, GrB_UINT16);
            }
        }
        32 => {
            quote! {
                #macro_identifier!(usize, GrB_UINT32);
            }
        }
        64 => {
            quote! {
                #macro_identifier!(usize, GrB_UINT64);
            }
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };

    TokenStream::from(expanded)
}

// #[proc_macro]
// pub fn

// https://blog.logrocket.com/macros-in-rust-a-tutorial-with-examples/#declarativemacrosinrust

// each macro can have multiple arms, matching its inputs
// Macros can be recursive, or calls to other macros

// NOTE: a proc-macro crate can only export proc-macros!
