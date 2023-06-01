extern crate proc_macro;
use proc_macro::{Ident, Span, TokenStream};
use quote::{format_ident, quote, TokenStreamExt};
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, Token};

// TODO: refactor this module to reduce code duplication

// #[proc_macro]
// pub fn graphblas_built_in_type_for_usize(_item: TokenStream) -> TokenStream {
//     match usize::BITS {
//         8 => "GrB_UINT8".parse().unwrap(),
//         16 => "GrB_UINT16".parse().unwrap(),
//         32 => "GrB_UINT32".parse().unwrap(),
//         64 => "GrB_UINT64".parse().unwrap(),
//         _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
//     }
// }

#[proc_macro]
pub fn graphblas_implementation_type_for_isize(_item: TokenStream) -> TokenStream {
    match isize::BITS {
        8 => "u8".parse().unwrap(),
        16 => "u16".parse().unwrap(),
        32 => "u32".parse().unwrap(),
        64 => "u64".parse().unwrap(),
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    }
}

#[proc_macro]
pub fn graphblas_implementation_type_for_usize(_item: TokenStream) -> TokenStream {
    match usize::BITS {
        8 => "u8".parse().unwrap(),
        16 => "u16".parse().unwrap(),
        32 => "u32".parse().unwrap(),
        64 => "u64".parse().unwrap(),
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    }
}

#[proc_macro]
pub fn graphblas_identifier_for_isize(input: TokenStream) -> TokenStream {
    let untyped_identifier = syn::parse_macro_input!(input as syn::Ident);
    let typed_identifier;
    match isize::BITS {
        8 => {
            typed_identifier = format_ident!("{}_INT8", untyped_identifier);
        }
        16 => {
            typed_identifier = format_ident!("{}_INT16", untyped_identifier);
        }
        32 => {
            typed_identifier = format_ident!("{}_INT32", untyped_identifier);
        }
        64 => {
            typed_identifier = format_ident!("{}_INT64", untyped_identifier);
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {#typed_identifier};
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn typed_graphblas_identifier(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let untyped_identifier = idents[0].to_owned();
    let value_type = idents[1].to_owned();

    let typed_identifier;
    match value_type.to_string().as_str() {
        "bool" => typed_identifier = format_ident!("{}_BOOL", untyped_identifier),
        "i8" => typed_identifier = format_ident!("{}_INT8", untyped_identifier),
        "i16" => typed_identifier = format_ident!("{}_INT16", untyped_identifier),
        "i32" => typed_identifier = format_ident!("{}_INT32", untyped_identifier),
        "i64" => typed_identifier = format_ident!("{}_INT64", untyped_identifier),
        "u8" => typed_identifier = format_ident!("{}_UINT8", untyped_identifier),
        "u16" => typed_identifier = format_ident!("{}_UINT16", untyped_identifier),
        "u32" => typed_identifier = format_ident!("{}_UINT32", untyped_identifier),
        "u64" => typed_identifier = format_ident!("{}_UINT64", untyped_identifier),
        "f32" => typed_identifier = format_ident!("{}_FP32", untyped_identifier),
        "f64" => typed_identifier = format_ident!("{}_FP64", untyped_identifier),
        "isize" => {
            match isize::BITS {
                8 => {
                    typed_identifier = format_ident!("{}_INT8", untyped_identifier);
                }
                16 => {
                    typed_identifier = format_ident!("{}_INT16", untyped_identifier);
                }
                32 => {
                    typed_identifier = format_ident!("{}_INT32", untyped_identifier);
                }
                64 => {
                    typed_identifier = format_ident!("{}_INT64", untyped_identifier);
                }
                _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
            };
        }
        "usize" => {
            match usize::BITS {
                8 => {
                    typed_identifier = format_ident!("{}_UINT8", untyped_identifier);
                }
                16 => {
                    typed_identifier = format_ident!("{}_UINT16", untyped_identifier);
                }
                32 => {
                    typed_identifier = format_ident!("{}_UINT32", untyped_identifier);
                }
                64 => {
                    typed_identifier = format_ident!("{}_UINT64", untyped_identifier);
                }
                _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
            };
        }
        _ => panic!("Unsupported value type: {:?}", value_type),
    }
    let expanded = quote! {#typed_identifier};
    TokenStream::from(expanded)
}

// pub(crate) fn typed_graphblas_identifier_from_ident(untyped_identifier: syn::Ident, value_type: syn::Ident) -> syn::Ident {
//     let typed_identifier = typed_graphblas_identifier(TokenStream::from(quote!(#untyped_identifier, #value_type)));
//     let typed_identifier_ = syn::parse_macro_input!(typed_identifier as syn::Ident);
//     return typed_identifier_
// }

// macro_rules! typed_graphblas_identifier_from_ident {
//     ($identifier: ident, $value_type: ident) => {
//         let token_stream = typed_graphblas_identifier(TokenStream::from(quote!($identifier, $value_type)));
//         $identifier = syn::parse_macro_input!($token_stream as syn::Ident);
//     };
// }

#[proc_macro]
pub fn graphblas_identifier_for_usize(input: TokenStream) -> TokenStream {
    graphblas_identifier_for_isize(input)
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
pub fn implement_macro(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let value_type = idents[0].to_owned();
    let matrix_reducer_operator = idents[1].to_owned();
    let vector_reducer_operator = idents[2].to_owned();

    // let matrix_reducer_operator = typed_graphblas_identifier_from_ident!(matrix_reducer_operator, value_type);
    // let matrix_reducer_operator = typed_graphblas_identifier_from_ident(matrix_reducer_operator, value_type.to_owned());
    let matrix_reducer_operator = typed_graphblas_identifier(TokenStream::from(
        quote!(#matrix_reducer_operator, #value_type),
    ));
    let matrix_reducer_operator = syn::parse_macro_input!(matrix_reducer_operator as syn::Ident);

    let mut expanded = quote! {};
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_monoid_reducer(input: TokenStream) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let value_type = idents[0].to_owned();
    let matrix_reducer_operator = idents[1].to_owned();
    let vector_reducer_operator = idents[2].to_owned();

    // let matrix_reducer_operator = typed_graphblas_identifier_from_ident!(matrix_reducer_operator, value_type);
    // let matrix_reducer_operator = typed_graphblas_identifier_from_ident(matrix_reducer_operator, value_type.to_owned());
    let matrix_reducer_operator = typed_graphblas_identifier(TokenStream::from(
        quote!(#matrix_reducer_operator, #value_type),
    ));
    let matrix_reducer_operator = syn::parse_macro_input!(matrix_reducer_operator as syn::Ident);

    let vector_reducer_operator = typed_graphblas_identifier(TokenStream::from(
        quote!(#vector_reducer_operator, #value_type),
    ));
    let vector_reducer_operator = syn::parse_macro_input!(vector_reducer_operator as syn::Ident);

    let mut expanded = quote! {
        impl MonoidScalarReducer<#value_type> for MonoidReducer<#value_type> {
            fn matrix_to_scalar(
                &self,
                argument: &SparseMatrix<#value_type>,
                product: &mut #value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();
                // let mut tmp_product = product.to_owned();
                // $convert_to_type!(tmp_product, $graphblas_implementation_type);

                context.call(|| unsafe {
                    #matrix_reducer_operator(
                        // &mut tmp_product,
                        product,
                        self.accumulator,
                        self.monoid,
                        argument.graphblas_matrix(),
                        self.options,
                    )
                })?;

                // $convert_to_type!(tmp_product, $value_type);
                // *product = tmp_product;
                Ok(())
            }

            fn vector_to_scalar(
                &self,
                argument: &SparseVector<#value_type>,
                product: &mut #value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();
                // let mut tmp_product = product.to_owned();
                // $convert_to_type!(tmp_product, $graphblas_implementation_type);

                context.call(|| unsafe {
                    #vector_reducer_operator(
                        // &mut tmp_product,
                        product,
                        self.accumulator,
                        self.monoid,
                        argument.graphblas_vector(),
                        self.options,
                    )
                })?;

                // $convert_to_type!(tmp_product, $value_type);
                // *product = tmp_product;
                Ok(())
            }
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_2_type_macro_for_isize_and_typed_graphblas_function(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(isize, isize, #function_identifier);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_2_type_macro_for_usize_and_typed_graphblas_function(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
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
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
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
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(usize, #function_identifier);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_isize_and_2_typed_graphblas_functions(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier_1 = idents[1].to_owned();
    let graphblas_function_identifier_2 = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(isize, #function_identifier_1, #function_identifier_2);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_usize_and_2_typed_graphblas_functions(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier_1 = idents[1].to_owned();
    let graphblas_function_identifier_2 = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(usize, #function_identifier_1, #function_identifier_2);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_isize_and_2_typed_graphblas_functions_with_type_conversion(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier_1 = idents[1].to_owned();
    let graphblas_function_identifier_2 = idents[2].to_owned();
    let type_conversion = idents[3].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_usize_and_2_typed_graphblas_functions_with_type_conversion(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier_1 = idents[1].to_owned();
    let graphblas_function_identifier_2 = idents[2].to_owned();
    let type_conversion = idents[3].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_isize_and_graphblas_function_with_type_conversion(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();
    let type_conversion = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_for_usize_and_graphblas_function_with_type_conversion(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();
    let type_conversion = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_1_type_macro_for_isize_and_typed_graphblas_function_with_implementation_type(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();

    let function_identifier;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, i8, #function_identifier);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, i16, #function_identifier);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, i32, #function_identifier);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, i64, #function_identifier);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_1_type_macro_for_usize_and_typed_graphblas_function_with_implementation_type(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();

    let function_identifier;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier = format_ident!("{}_UINT8", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, u8, #function_identifier);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_UINT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, u16, #function_identifier);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_UINT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, u32, #function_identifier);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_UINT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(usize, u64, #function_identifier);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_1_type_macro_for_isize_and_2_typed_graphblas_functions_with_implementation_type(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier_1 = idents[1].to_owned();
    let graphblas_function_identifier_2 = idents[2].to_owned();

    let function_identifier_1;
    let function_identifier_2;
    let expanded;
    match isize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_INT8", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT8", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(isize, i8, #function_identifier_1, #function_identifier_2);
            };
        }
        16 => {
            function_identifier_1 = format_ident!("{}_INT16", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT16", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(isize, i16, #function_identifier_1, #function_identifier_2);
            };
        }
        32 => {
            function_identifier_1 = format_ident!("{}_UINT32", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT32", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(isize, i32, #function_identifier_1, #function_identifier_2);
            };
        }
        64 => {
            function_identifier_1 = format_ident!("{}_INT64", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT64", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(isize, i64, #function_identifier_1, #function_identifier_2);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_1_type_macro_for_usize_and_2_typed_graphblas_functions_with_implementation_type(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier_1 = idents[1].to_owned();
    let graphblas_function_identifier_2 = idents[2].to_owned();

    let function_identifier_1;
    let function_identifier_2;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_UINT8", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT8", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(usize, u8, #function_identifier_1, #function_identifier_2);
            };
        }
        16 => {
            function_identifier_1 = format_ident!("{}_UINT16", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT16", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(usize, u16, #function_identifier_1, #function_identifier_2);
            };
        }
        32 => {
            function_identifier_1 = format_ident!("{}_UINT32", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT32", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(usize, u32, #function_identifier_1, #function_identifier_2);
            };
        }
        64 => {
            function_identifier_1 = format_ident!("{}_UINT64", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT64", graphblas_function_identifier_2);
            expanded = quote! {
                #macro_identifier!(usize, u64, #function_identifier_1, #function_identifier_2);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_1_type_macro_for_isize_and_4_typed_graphblas_functions_with_implementation_type(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier_1 = idents[1].to_owned();
    let graphblas_function_identifier_2 = idents[2].to_owned();
    let graphblas_function_identifier_3 = idents[3].to_owned();
    let graphblas_function_identifier_4 = idents[4].to_owned();

    let function_identifier_1;
    let function_identifier_2;
    let function_identifier_3;
    let function_identifier_4;
    let expanded;
    match isize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_INT8", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT8", graphblas_function_identifier_2);
            function_identifier_3 = format_ident!("{}_INT8", graphblas_function_identifier_3);
            function_identifier_4 = format_ident!("{}_INT8", graphblas_function_identifier_4);
            expanded = quote! {
                #macro_identifier!(isize, i8, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        16 => {
            function_identifier_1 = format_ident!("{}_INT16", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT16", graphblas_function_identifier_2);
            function_identifier_3 = format_ident!("{}_INT16", graphblas_function_identifier_3);
            function_identifier_4 = format_ident!("{}_INT16", graphblas_function_identifier_4);
            expanded = quote! {
                #macro_identifier!(isize, i16, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        32 => {
            function_identifier_1 = format_ident!("{}_UINT32", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT32", graphblas_function_identifier_2);
            function_identifier_3 = format_ident!("{}_UINT32", graphblas_function_identifier_3);
            function_identifier_4 = format_ident!("{}_UINT32", graphblas_function_identifier_4);
            expanded = quote! {
                #macro_identifier!(isize, i32, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        64 => {
            function_identifier_1 = format_ident!("{}_INT64", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_INT64", graphblas_function_identifier_2);
            function_identifier_3 = format_ident!("{}_INT64", graphblas_function_identifier_3);
            function_identifier_4 = format_ident!("{}_INT64", graphblas_function_identifier_4);
            expanded = quote! {
                #macro_identifier!(isize, i64, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_1_type_macro_for_usize_and_4_typed_graphblas_functions_with_implementation_type(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier_1 = idents[1].to_owned();
    let graphblas_function_identifier_2 = idents[2].to_owned();
    let graphblas_function_identifier_3 = idents[3].to_owned();
    let graphblas_function_identifier_4 = idents[4].to_owned();

    let function_identifier_1;
    let function_identifier_2;
    let function_identifier_3;
    let function_identifier_4;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier_1 = format_ident!("{}_UINT8", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT8", graphblas_function_identifier_2);
            function_identifier_3 = format_ident!("{}_UINT8", graphblas_function_identifier_3);
            function_identifier_4 = format_ident!("{}_UINT8", graphblas_function_identifier_4);
            expanded = quote! {
                #macro_identifier!(usize, u8, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        16 => {
            function_identifier_1 = format_ident!("{}_UINT16", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT16", graphblas_function_identifier_2);
            function_identifier_3 = format_ident!("{}_UINT16", graphblas_function_identifier_3);
            function_identifier_4 = format_ident!("{}_UINT16", graphblas_function_identifier_4);
            expanded = quote! {
                #macro_identifier!(usize, u16, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        32 => {
            function_identifier_1 = format_ident!("{}_UINT32", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT32", graphblas_function_identifier_2);
            function_identifier_3 = format_ident!("{}_UINT32", graphblas_function_identifier_3);
            function_identifier_4 = format_ident!("{}_UINT32", graphblas_function_identifier_4);
            expanded = quote! {
                #macro_identifier!(usize, u32, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        64 => {
            function_identifier_1 = format_ident!("{}_UINT64", graphblas_function_identifier_1);
            function_identifier_2 = format_ident!("{}_UINT64", graphblas_function_identifier_2);
            function_identifier_3 = format_ident!("{}_UINT64", graphblas_function_identifier_3);
            function_identifier_4 = format_ident!("{}_UINT64", graphblas_function_identifier_4);
            expanded = quote! {
                #macro_identifier!(usize, u64, #function_identifier_1, #function_identifier_2, #function_identifier_3, #function_identifier_4);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_2_type_macro_for_isize_and_typed_graphblas_function_with_type_conversion(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();
    let type_conversion = idents[2].to_owned();

    let function_identifier;
    let expanded;
    match usize::BITS {
        8 => {
            function_identifier = format_ident!("{}_INT8", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, isize, i8, #function_identifier, #type_conversion);
            };
        }
        16 => {
            function_identifier = format_ident!("{}_INT16", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, isize, i16, #function_identifier, #type_conversion);
            };
        }
        32 => {
            function_identifier = format_ident!("{}_INT32", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, isize, i32, #function_identifier, #type_conversion);
            };
        }
        64 => {
            function_identifier = format_ident!("{}_INT64", graphblas_function_identifier);
            expanded = quote! {
                #macro_identifier!(isize, isize, i64, #function_identifier, #type_conversion);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_2_type_macro_for_usize_and_typed_graphblas_function_with_type_conversion(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_identifier = idents[1].to_owned();
    let type_conversion = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
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
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_1 = idents[1].to_owned();
    let graphblas_function_2 = idents[2].to_owned();
    let graphblas_function_3 = idents[3].to_owned();
    let graphblas_function_4 = idents[4].to_owned();
    let type_conversion = idents[5].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
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
    let macro_identifier = idents[0].to_owned();
    let graphblas_function_1 = idents[1].to_owned();
    let graphblas_function_2 = idents[2].to_owned();
    let graphblas_function_3 = idents[3].to_owned();
    let graphblas_function_4 = idents[4].to_owned();
    let type_conversion = idents[5].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, isize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, usize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_1_type_trait_and_typed_graphblas_function_for_isize_with_postfix(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();
    let postfix = idents[3].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, isize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_1_type_trait_and_typed_graphblas_function_for_usize_with_postfix(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();
    let postfix = idents[3].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, usize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_3_typed_trait_and_typed_graphblas_function_for_isize(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, isize, isize, isize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_2_typed_trait_and_typed_graphblas_function_for_usize(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, usize, usize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_2_typed_trait_and_typed_graphblas_function_for_isize(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, isize, isize);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_2_typed_trait_and_output_type_and_typed_graphblas_function_for_usize(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();
    let output_type = idents[3].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, usize, usize, #output_type);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_2_typed_trait_and_output_type_and_typed_graphblas_function_for_isize(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();
    let output_type = idents[3].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#trait_identifier, #function_identifier, isize, isize, #output_type);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_macro_with_3_typed_trait_and_typed_graphblas_function_for_usize(
    input: TokenStream,
) -> TokenStream {
    // TODO: develop a more structured, reliable, and self-documenting way to parse the input.
    // Maybe by defining a data struct, and implementing Parse?
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let trait_identifier = idents[1].to_owned();
    let graphblas_function = idents[2].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
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
    let macro_identifier = idents[0].to_owned();
    let semiring = idents[1].to_owned();
    let addition_operator = idents[2].to_owned();
    let multiplication_operator = idents[3].to_owned();
    let graphblas_function = idents[4].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#semiring, #addition_operator, #multiplication_operator, isize, isize, isize, isize, #function_identifier);
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
    let macro_identifier = idents[0].to_owned();
    let semiring = idents[1].to_owned();
    let addition_operator = idents[2].to_owned();
    let multiplication_operator = idents[3].to_owned();
    let graphblas_function = idents[4].to_owned();

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
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    let expanded = quote! {
        #macro_identifier!(#semiring, #addition_operator, #multiplication_operator, usize, usize, usize, usize, #function_identifier);
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_type_conversion_macro_for_isize(input: TokenStream) -> TokenStream {
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let type_conversion_macro = idents[1].to_owned();

    let expanded;
    match usize::BITS {
        8 => {
            expanded = quote! {
                #macro_identifier!(isize, i8, #type_conversion_macro);
                #macro_identifier!(i8, isize, #type_conversion_macro);
            };
        }
        16 => {
            expanded = quote! {
                #macro_identifier!(isize, i16, #type_conversion_macro);
                #macro_identifier!(i16, isize, #type_conversion_macro);
            };
        }
        32 => {
            expanded = quote! {
                #macro_identifier!(isize, i32, #type_conversion_macro);
                #macro_identifier!(i32, isize, #type_conversion_macro);
            };
        }
        64 => {
            expanded = quote! {
                #macro_identifier!(isize, i64, #type_conversion_macro);
                #macro_identifier!(i64, isize, #type_conversion_macro);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", isize::BITS),
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn implement_type_conversion_macro_for_usize(input: TokenStream) -> TokenStream {
    let idents: Vec<syn::Ident> = parse_macro_input!(
        input with Punctuated::<syn::Ident, Token![,]>::parse_terminated)
    .into_iter()
    .collect();
    let macro_identifier = idents[0].to_owned();
    let type_conversion_macro = idents[1].to_owned();

    let expanded;
    match usize::BITS {
        8 => {
            expanded = quote! {
                #macro_identifier!(usize, u8, #type_conversion_macro);
                #macro_identifier!(u8, usize, #type_conversion_macro);
            };
        }
        16 => {
            expanded = quote! {
                #macro_identifier!(usize, u16, #type_conversion_macro);
                #macro_identifier!(u16, usize, #type_conversion_macro);
            };
        }
        32 => {
            expanded = quote! {
                #macro_identifier!(usize, u32, #type_conversion_macro);
                #macro_identifier!(u32, usize, #type_conversion_macro);
            };
        }
        64 => {
            expanded = quote! {
                #macro_identifier!(usize, u64, #type_conversion_macro);
                #macro_identifier!(u64, usize, #type_conversion_macro);
            };
        }
        _ => panic!("Unsupported architecture: {:?} bits", usize::BITS),
    };
    TokenStream::from(expanded)
}

// NOTE: a proc-macro crate can only export proc-macros!
// https://blog.jetbrains.com/rust/2022/03/18/procedural-macros-under-the-hood-part-i/
// test by "RUSTFLAGS="-Z macro-backtrace" cargo +nightly test"
