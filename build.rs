use std::collections::HashSet;
use std::env;
use std::ffi::OsString;
use std::path::PathBuf;

extern crate bindgen;
extern crate cmake;

#[derive(Debug)]
struct IgnoreMacros(HashSet<String>);

impl bindgen::callbacks::ParseCallbacks for IgnoreMacros {
    fn will_parse_macro(&self, name: &str) -> bindgen::callbacks::MacroParsingBehavior {
        if self.0.contains(name) {
            bindgen::callbacks::MacroParsingBehavior::Ignore
        } else {
            bindgen::callbacks::MacroParsingBehavior::Default
        }
    }
}

fn main() {
    if let Ok(_) = std::env::var("DOCS_RS") {
        // do not build any dependencies as this would time out the docs.rs build server
    } else {
        #[cfg(feature = "build_static_graphblas_dependencies")]
        build_and_link_dependencies();

        #[cfg(feature = "generate_new_bindings_to_graphblas_implementation")]
        generate_bindings_to_graphblas_implementation();
    }
}

fn path_with_graphblas_header_file() -> PathBuf {
    let mut path_with_graphblas_header_file = path_with_graphblas_implementation();
    path_with_graphblas_header_file.push("SuiteSparse_GraphBLAS");
    path_with_graphblas_header_file.push("Include");
    path_with_graphblas_header_file.push("GraphBLAS.h");
    return path_with_graphblas_header_file;
}

fn path_with_graphblas_implementation() -> PathBuf {
    let mut path_with_graphblas_implementation = std::env::current_dir().unwrap();
    path_with_graphblas_implementation.push("graphblas_implementation");
    return path_with_graphblas_implementation;
}

#[cfg(feature = "build_static_graphblas_dependencies")]
fn build_and_link_dependencies() {
    let cargo_build_directory = env::var_os("OUT_DIR").unwrap();
    let path_with_graphblas_header_file = path_with_graphblas_header_file();
    let path_with_graphblas_implementation = path_with_graphblas_implementation();

    #[cfg(feature = "build_static_graphblas_dependencies")]
    build_static_graphblas_implementation(&cargo_build_directory);

    // add directory containing libgomp.a to search path.
    // This directory should also contain libgraphblas.a, if it is not build from source or otherwise in the search path
    println!(
        "cargo:rustc-link-search=native={}",
        path_with_graphblas_implementation
            .clone()
            .to_str()
            .unwrap()
            .to_owned()
    );

    println!("cargo:rustc-link-lib=static=graphblas");
    println!("cargo:rustc-link-lib=static=gomp");

    declare_build_invalidation_conditions(
        &path_with_graphblas_implementation,
        &path_with_graphblas_header_file,
    );
}

// #[cfg(feature = "build_static_graphblas_dependencies")]
fn build_static_graphblas_implementation(cargo_build_directory: &OsString) {
    let _dst = cmake::Config::new("graphblas_implementation/SuiteSparse_GraphBLAS")
        .define("JOBS", "32")
        .define("BUILD_GRB_STATIC_LIBRARY", "true")
        .define("CMAKE_INSTALL_LIBDIR", cargo_build_directory.clone())
        .define("CMAKE_INSTALL_INCLUDEDIR", cargo_build_directory.clone())
        .define("PROJECT_SOURCE_DIR", cargo_build_directory.clone()) // prevent modifying config files outside of the cargo output directory
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        cargo_build_directory.clone().to_str().unwrap().to_owned()
    );
}

fn declare_build_invalidation_conditions(
    path_with_graphblas_implementation: &PathBuf,
    path_with_graphblas_header_file: &PathBuf,
) {
    println!(
        "cargo:rerun-if-changed = {}",
        path_with_graphblas_implementation
            .clone()
            .join("wrapper.h")
            .to_str()
            .unwrap()
            .to_owned()
    );
    println!(
        "cargo:rerun-if-changed = {}",
        path_with_graphblas_header_file
            .clone()
            .to_str()
            .unwrap()
            .to_owned()
    );
    println!(
        "cargo:rerun-if-changed = {}",
        path_with_graphblas_implementation
            .clone()
            .join("libgomp.a")
            .to_str()
            .unwrap()
            .to_owned()
    );
    println!("cargo:rerun-if-changed=build.rs");
}

fn generate_bindings_to_graphblas_implementation() {
    let path_with_graphblas_header_file = path_with_graphblas_header_file();
    let path_with_graphblas_implementation = path_with_graphblas_implementation();

    let ignored_macros = IgnoreMacros(
        vec![
            "FP_INFINITE".into(),
            "FP_NAN".into(),
            "FP_NORMAL".into(),
            "FP_SUBNORMAL".into(),
            "FP_ZERO".into(),
            "IPPORT_RESERVED".into(),
        ]
        .into_iter()
        .collect(),
    );

    let bindings = bindgen::Builder::default()
        .header(
            path_with_graphblas_header_file
                .clone()
                .to_str()
                .unwrap()
                .to_owned(),
        )
        .parse_callbacks(Box::new(ignored_macros))
        .generate()
        .expect("Unable to generate bindings");

    let mut bindings_target_path = path_with_graphblas_implementation.clone();
    bindings_target_path.push("suitesparse_graphblas_bindings.rs");
    bindings
        .write_to_file(bindings_target_path.to_str().unwrap().to_owned())
        .expect("Couldn't write bindings!");
}
