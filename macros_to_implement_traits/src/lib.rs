extern crate proc_macro;
use proc_macro::{TokenStream};
use quote::{format_ident, quote};
use syn::{parse_macro_input, Token};
use syn::punctuated::Punctuated;

// TODO: refactor this module to reduce code duplication

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

#[proc_macro]
pub fn implement_2_type_macro_for_isize_and_typed_graphblas_function(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier = idents[1].clone();
    
    let function_identifier;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function_identifier);
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function_identifier);
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function_identifier);
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function_identifier);
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(isize, isize, #function_identifier);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_2_type_macro_for_usize_and_typed_graphblas_function(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier = idents[1].clone();
    
    let function_identifier;
    match usize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8", graphblas_function_identifier);
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function_identifier);
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function_identifier);
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function_identifier);
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(usize, usize, #function_identifier);
    };
    TokenStream::from(expanded)
}

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
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function_identifier);
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function_identifier);
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function_identifier);
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function_identifier);
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(isize, #function_identifier);
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
    match usize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8", graphblas_function_identifier);
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function_identifier);
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function_identifier);
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function_identifier);
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(usize, #function_identifier);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_isize_and_2_typed_graphblas_functions(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier_1 = idents[1].clone();
    let graphblas_function_identifier_2 = idents[2].clone();
    
    let function_identifier_1;
    let function_identifier_2;
    match isize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_INT8", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT8", graphblas_function_identifier_2);
        }
        16 => {
            function_identifier_1 = format_ident!("{}_INT16", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT16", graphblas_function_identifier_2);
        }
        32 => {
            function_identifier_1 = format_ident!("{}_INT32", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT32", graphblas_function_identifier_2);
        }
        64 => {
            function_identifier_1 = format_ident!("{}_INT64", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT64", graphblas_function_identifier_2);
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(isize, #function_identifier_1, #function_identifier_2);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_usize_and_2_typed_graphblas_functions(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier_1 = idents[1].clone();
    let graphblas_function_identifier_2 = idents[2].clone();
    
    let function_identifier_1;
    let function_identifier_2;
    match usize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_UINT8", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT8", graphblas_function_identifier_2);
        }
        16 => {
            function_identifier_1 = format_ident!("{}_UINT16", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT16", graphblas_function_identifier_2);
        }
        32 => {
            function_identifier_1 = format_ident!("{}_UINT32", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT32", graphblas_function_identifier_2);
        }
        64 => {
            function_identifier_1 = format_ident!("{}_UINT64", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT64", graphblas_function_identifier_2);
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(usize, #function_identifier_1, #function_identifier_2);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_isize_and_2_typed_graphblas_functions_with_type_conversion(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier_1 = idents[1].clone();
    let graphblas_function_identifier_2 = idents[2].clone();
    let type_conversion = idents[3].clone();
    
    let function_identifier_1;
    let function_identifier_2;
    let expanded;
    match isize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_INT8", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT8", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(isize, i8, #function_identifier_1, #function_identifier_2, #type_conversion);
            };
        }
        16 => {
            function_identifier_1 = format_ident!("{}_INT16", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT16", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(isize, i16, #function_identifier_1, #function_identifier_2, #type_conversion);
            };
        }
        32 => {
            function_identifier_1 = format_ident!("{}_INT32", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT32", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(isize, i32, #function_identifier_1, #function_identifier_2, #type_conversion);
            };
        }
        64 => {
            function_identifier_1 = format_ident!("{}_INT64", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT64", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(isize, i64, #function_identifier_1, #function_identifier_2, #type_conversion);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_usize_and_2_typed_graphblas_functions_with_type_conversion(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_identifier_1 = idents[1].clone();
    let graphblas_function_identifier_2 = idents[2].clone();
    let type_conversion = idents[3].clone();
    
    let function_identifier_1;
    let function_identifier_2;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_UINT8", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT8", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(usize, u8, #function_identifier_1, #function_identifier_2, #type_conversion);
            };
        }
        16 => {
            function_identifier_1 = format_ident!("{}_UINT16", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT16", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(usize, u16, #function_identifier_1, #function_identifier_2, #type_conversion);
            };
        }
        32 => {
            function_identifier_1 = format_ident!("{}_UINT32", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT32", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(usize, u32, #function_identifier_1, #function_identifier_2, #type_conversion);
            };
        }
        64 => {
            function_identifier_1 = format_ident!("{}_UINT64", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT64", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(usize, u64, #function_identifier_1, #function_identifier_2, #type_conversion);
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

#[proc_macro]
pub fn implement_2_type_macro_for_isize_and_typed_graphblas_function_with_type_conversion(input: TokenStream) -> TokenStream {
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
                #macro_identifier!(isize, isize, u8, #function_identifier, #type_conversion);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, isize, u16, #function_identifier, #type_conversion);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, isize, u32, #function_identifier, #type_conversion);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, isize, u64, #function_identifier, #type_conversion);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_2_type_macro_for_usize_and_typed_graphblas_function_with_type_conversion(input: TokenStream) -> TokenStream {
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
                #macro_identifier!(usize, usize, u8, #function_identifier, #type_conversion);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, usize, u16, #function_identifier, #type_conversion);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, usize, u32, #function_identifier, #type_conversion);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, usize, u64, #function_identifier, #type_conversion);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_3_isizes_and_4_graphblas_functions(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_1 = idents[1].clone();
    let graphblas_function_2 = idents[2].clone();
    let graphblas_function_3 = idents[3].clone();
    let graphblas_function_4 = idents[4].clone();
    let type_conversion = idents[5].clone();
    
    let function_identifier_1;
    let function_identifier_2;
    let function_identifier_3;
    let function_identifier_4;
    let expanded;
    match isize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_INT8", graphblas_function_1);
            function_identifier_2 = format_ident!("{}_INT8", graphblas_function_2);
            function_identifier_3 = format_ident!("{}_INT8", graphblas_function_3);
            function_identifier_4 = format_ident!("{}_INT8", graphblas_function_4);
            expanded = quote! {
                #macro_identifier!(isize, isize, isize, i8, #type_conversion, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        16 => {
            function_identifier_1 = format_ident!("{}_INT16", graphblas_function_1);
            function_identifier_2 = format_ident!("{}_INT16", graphblas_function_2);
            function_identifier_3 = format_ident!("{}_INT16", graphblas_function_3);
            function_identifier_4 = format_ident!("{}_INT16", graphblas_function_4);
            expanded = quote! {
                #macro_identifier!(isize, isize, isize, i16, #type_conversion, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        32 => {
            function_identifier_1 = format_ident!("{}_INT32", graphblas_function_1);
            function_identifier_2 = format_ident!("{}_INT32", graphblas_function_2);
            function_identifier_3 = format_ident!("{}_INT32", graphblas_function_3);
            function_identifier_4 = format_ident!("{}_INT32", graphblas_function_4);
            expanded = quote! {
                #macro_identifier!(isize, isize, isize, i32, #type_conversion, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        64 => {
            function_identifier_1 = format_ident!("{}_INT64", graphblas_function_1);
            function_identifier_2 = format_ident!("{}_INT64", graphblas_function_2);
            function_identifier_3 = format_ident!("{}_INT64", graphblas_function_3);
            function_identifier_4 = format_ident!("{}_INT64", graphblas_function_4);
            expanded = quote! {
                #macro_identifier!(isize, isize, isize, i64, #type_conversion, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    TokenStream::from(expanded)
}

// macro_rules! to_u8_typed_graphblas_function {
//     (graphblas_function:ident) => {
//         format_ident!("{}_UINT8", graphblas_function)
//     };
// }

#[proc_macro]
pub fn implement_macro_for_3_usizes_and_4_graphblas_functions(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let graphblas_function_1 = idents[1].clone();
    let graphblas_function_2 = idents[2].clone();
    let graphblas_function_3 = idents[3].clone();
    let graphblas_function_4 = idents[4].clone();
    let type_conversion = idents[5].clone();
    
    let function_identifier_1;
    let function_identifier_2;
    let function_identifier_3;
    let function_identifier_4;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_UINT8", graphblas_function_1);
            // function_identifier_1 = quote!{to_u8_typed_graphblas_function!(graphblas_function_1)};
            function_identifier_2 = format_ident!("{}_UINT8", graphblas_function_2);
            function_identifier_3 = format_ident!("{}_UINT8", graphblas_function_3);
            function_identifier_4 = format_ident!("{}_UINT8", graphblas_function_4);
            expanded = quote! {
                #macro_identifier!(usize, usize, usize, u8, #type_conversion, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        16 => {
            function_identifier_1 = format_ident!("{}_UINT16", graphblas_function_1);
            function_identifier_2 = format_ident!("{}_UINT16", graphblas_function_2);
            function_identifier_3 = format_ident!("{}_UINT16", graphblas_function_3);
            function_identifier_4 = format_ident!("{}_UINT16", graphblas_function_4);
            expanded = quote! {
                #macro_identifier!(usize, usize, usize, u16, #type_conversion, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        32 => {
            function_identifier_1 = format_ident!("{}_UINT32", graphblas_function_1);
            function_identifier_2 = format_ident!("{}_UINT32", graphblas_function_2);
            function_identifier_3 = format_ident!("{}_UINT32", graphblas_function_3);
            function_identifier_4 = format_ident!("{}_UINT32", graphblas_function_4);
            expanded = quote! {
                #macro_identifier!(usize, usize, usize, u32, #type_conversion, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        64 => {
            function_identifier_1 = format_ident!("{}_UINT64", graphblas_function_1);
            function_identifier_2 = format_ident!("{}_UINT64", graphblas_function_2);
            function_identifier_3 = format_ident!("{}_UINT64", graphblas_function_3);
            function_identifier_4 = format_ident!("{}_UINT64", graphblas_function_4);
            expanded = quote! {
                #macro_identifier!(usize, usize, usize, u64, #type_conversion, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let trait_identifier  = idents[1].clone();
    let graphblas_function = idents[2].clone();
    
    let function_identifier;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function);
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function);
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function);
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function);
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, isize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let trait_identifier  = idents[1].clone();
    let graphblas_function = idents[2].clone();
    
    let function_identifier;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8", graphblas_function);
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function);
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function);
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function);
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, usize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize_with_postfix(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let trait_identifier  = idents[1].clone();
    let graphblas_function = idents[2].clone();
    let postfix = idents[3].clone();
    
    let function_identifier;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8_{}", graphblas_function, postfix);
        }
        16 => {
            function_identifier = format_ident!("{}_INT16_{}", graphblas_function, postfix);
        }
        32 => {
            function_identifier = format_ident!("{}_INT32_{}", graphblas_function, postfix);
        }
        64 => {
            function_identifier = format_ident!("{}_INT64_{}", graphblas_function, postfix);
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, isize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize_with_postfix(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let trait_identifier  = idents[1].clone();
    let graphblas_function = idents[2].clone();
    let postfix = idents[3].clone();
    
    let function_identifier;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8_{}", graphblas_function, postfix);
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16_{}", graphblas_function, postfix);
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32_{}", graphblas_function, postfix);
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64_{}", graphblas_function, postfix);
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, usize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_3_typed_trait_and_typed_graphblas_function_for_isize(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let trait_identifier  = idents[1].clone();
    let graphblas_function = idents[2].clone();
    
    let function_identifier;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function);
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function);
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function);
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function);
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, isize, isize, isize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_3_typed_trait_and_typed_graphblas_function_for_usize(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let trait_identifier  = idents[1].clone();
    let graphblas_function = idents[2].clone();
    
    let function_identifier;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8", graphblas_function);
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function);
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function);
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function);
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, usize, usize, usize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_semiring_for_isize(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let semiring = idents[1].clone();
    let addition_operator = idents[2].clone();
    let multiplication_operator = idents[3].clone();
    let graphblas_function = idents[4].clone();
    
    let function_identifier;
    match isize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function);
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function);
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function);
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function);
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(#semiring, #addition_operator, #multiplication_operator, isize, isize, isize, #function_identifier);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_semiring_for_usize(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
        .into_iter()
        .collect();
    let macro_identifier = idents[0].clone();
    let semiring = idents[1].clone();
    let addition_operator = idents[2].clone();
    let multiplication_operator = idents[3].clone();
    let graphblas_function = idents[4].clone();

    let function_identifier;
    match usize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8", graphblas_function);
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function);
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function);
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function);
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS)
    };
    let expanded = quote! {
        #macro_identifier!(#semiring, #addition_operator, #multiplication_operator, usize, usize, usize, #function_identifier);
    };
    TokenStream::from(expanded)
}

// NOTE: a proc-macro crate can only export proc-macros!

// https://blog.jetbrains.com/rust/2022/03/18/procedural-macros-under-the-hood-part-i/

// test by "RUSTFLAGS="-Z macro-backtrace" cargo +nightly test"