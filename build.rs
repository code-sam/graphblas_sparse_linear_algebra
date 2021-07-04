use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};

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

// https://kornel.ski/rust-sys-crate

// NOTE: update linker cache before running: (when using dynamic linking to shared object)
// https://itsfoss.com/solve-open-shared-object-file-quick-tip/
// "sudo /sbin/ldconfig -v"

fn main() {
    let mut path_with_graphblas_implementation = std::env::current_dir().unwrap();
    path_with_graphblas_implementation.push("graphblas_implementation");
    let graphblas_build_target_path = path_with_graphblas_implementation
        .clone()
        .to_str()
        .unwrap()
        .to_owned();

    let mut dst = cmake::Config::new("graphblas_implementation/SuiteSparse_GraphBLAS")
        .define("JOBS", "32")
        .define("BUILD_GRB_STATIC_LIBRARY", "true")
        .define("CMAKE_INSTALL_LIBDIR", graphblas_build_target_path.clone())
        .define(
            "CMAKE_INSTALL_INCLUDEDIR",
            graphblas_build_target_path.clone(),
        )
        .build();

    // println!("cargo:rustc-link-search=native={}", dst.display());
    // let directory_with_build_output = env::var_os("OUT_DIR").unwrap();
    // println!("cargo:rustc-link-search=native={}", directory_with_build_output.to_str().unwrap().to_owned());

    let mut path_with_graphblas_header_file = path_with_graphblas_implementation.clone();
    // path_with_graphblas_header_file.push("graphblas_implementation");
    // path_with_graphblas_header_file.push("SuiteSparse_GraphBLAS");
    // path_with_graphblas_header_file.push("Include");
    path_with_graphblas_header_file.push("GraphBLAS.h");
    //
    // Tell cargo to tell rustc to link the system GrapBLAS
    // shared library.
    println!(
        "cargo:rustc-link-search=native={}",
        graphblas_build_target_path.clone()
    );
    println!("cargo:rustc-link-lib=static=graphblas");

    // TODO: find directory with openMP archive gomp automatically
    println!("cargo:rustc-link-search=native=/usr/lib/gcc/x86_64-linux-gnu/7/");
    println!("cargo:rustc-link-lib=gomp");
    // println!("cargo:rustc-link-lib=static=gomp");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
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
        path_with_graphblas_implementation
            .clone()
            .join("GraphBLAS.h")
            .to_str()
            .unwrap()
            .to_owned()
    );
    let mut path_with_graphblas_implementation_header_file = path_with_graphblas_implementation.clone();
    path_with_graphblas_implementation_header_file.push("SuiteSparse_GraphBLAS");
    path_with_graphblas_implementation_header_file.push("Include");
    path_with_graphblas_implementation_header_file.push("GraphBLAS.h");
    // println!(
    //     "cargo:rerun-if-changed = {}",
    //     path_with_graphblas_implementation
    //         .clone()
    //         .join("libgraphblas.a")
    //         .to_str()
    //         .unwrap()
    //         .to_owned()
    // );
    println!("cargo:rerun-if-changed=build.rs");

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

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(
            path_with_graphblas_header_file
                .clone()
                .to_str()
                .unwrap()
                .to_owned(),
        )
        // .header(
        //     path_with_graphblas_implementation
        //         .clone()
        //         .join("wrapper.h")
        //         .to_str()
        //         .unwrap()
        //         .to_owned(),
        // )
        // .header("GraphBLAS/wrapper.h")
        .parse_callbacks(Box::new(ignored_macros))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        // .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the bindings.rs file.
    let mut bindings_target_path = std::env::current_dir().unwrap();
    bindings_target_path.push("graphblas_implementation");
    bindings_target_path.push("bindings.rs");
    bindings
        .write_to_file(bindings_target_path.to_str().unwrap().to_owned())
        .expect("Couldn't write bindings!");
}

// https://github.com/gcc-mirror/gcc/tree/master/libgomp
