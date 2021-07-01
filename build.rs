extern crate bindgen;

use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

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
    // Tell cargo to tell rustc to link the system GrapBLAS
    // shared library.
    println!(
        "cargo:rustc-link-search=native={}",
        path_with_graphblas_implementation
            .clone()
            .to_str()
            .unwrap()
            .to_owned()
    );
    // TODO: find directory with openMP archive gomp automatically
    println!("cargo:rustc-link-search=native=/usr/lib/gcc/x86_64-linux-gnu/7/");

    println!("cargo:rustc-link-lib=static=graphblas");
    println!("cargo:rustc-link-lib=static=gomp");

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
    println!(
        "cargo:rerun-if-changed = {}",
        path_with_graphblas_implementation
            .clone()
            .join("libgraphblas.a")
            .to_str()
            .unwrap()
            .to_owned()
    );

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
            path_with_graphblas_implementation
                .clone()
                .join("wrapper.h")
                .to_str()
                .unwrap()
                .to_owned(),
        )
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
    bindings_target_path.push("src");
    bindings_target_path.push("bindings_to_graphblas_implementation");
    bindings_target_path.push("bindings.rs");
    bindings
        .write_to_file(bindings_target_path.to_str().unwrap().to_owned())
        .expect("Couldn't write bindings!");
}
