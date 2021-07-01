#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// TODO: blacklist u128 API, or automatically check that it is not being used
#![allow(improper_ctypes)]
// TODO: only expose what is actually used
#![allow(dead_code)]

include!("../../graphblas_implementation/bindings.rs");
