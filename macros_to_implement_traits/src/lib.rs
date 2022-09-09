extern crate proc_macro;
use proc_macro::{TokenStream, TokenTree, Ident};
use quote::{format_ident, quote};
use syn::{parse_macro_input, DeriveInput, Token};
use syn::punctuated::Punctuated;

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

// TODO: refactor to reduce code duplication

// TODO: move all calls to implement_macro_for_isize_and_graphblas_function_with_type_conversion
#[proc_macro]
pub fn implement_macro_for_isize_and_graphblas_function(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier = idents[1].clone();
    
    let function_identifier;
    let expanded;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, #function_identifier);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, #function_identifier);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, #function_identifier);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, #function_identifier);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };

    TokenStream::from(expanded)
}

// TODO: move all calls to implement_macro_for_usize_and_graphblas_function_with_type_conversion
#[proc_macro]
pub fn implement_macro_for_usize_and_graphblas_function(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier = idents[1].clone();
    
    let function_identifier;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, #function_identifier);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, #function_identifier);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, #function_identifier);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, #function_identifier);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };

    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_isize_and_graphblas_function_with_type_conversion(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier = idents[1].clone();
    let type_conversion = idents[2].clone();
    
    let function_identifier;
    let expanded;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, i8, #function_identifier, #type_conversion);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, i16, #function_identifier, #type_conversion);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, i32, #function_identifier, #type_conversion);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, i64, #function_identifier, #type_conversion);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };

    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_usize_and_graphblas_function_with_type_conversion(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier = idents[1].clone();
    let type_conversion = idents[2].clone();
    
    let function_identifier;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, u8, #function_identifier, #type_conversion);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, u16, #function_identifier, #type_conversion);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, u32, #function_identifier, #type_conversion);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, u64, #function_identifier, #type_conversion);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };

    TokenStream::from(expanded)
}

// NOTE: a proc-macro crate can only export proc-macros!

// https://blog.jetbrains.com/rust/2022/03/18/procedural-macros-under-the-hood-part-i/

// test by "RUSTFLAGS="-Z macro-backtrace" cargo +nightly test"