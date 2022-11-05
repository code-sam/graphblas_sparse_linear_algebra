use std::collections::HashSet;
use std::{env, error};
use std::ffi::OsString;
use std::fs;
use std::path::{PathBuf};

use git2::{Object, Oid, Repository};
use glob::glob;

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
        build_and_link_dependencies();
        generate_bindings_to_graphblas_implementation();
    }
}

fn path_with_graphblas_implementation() -> PathBuf {
    // TODO: is it correct to reference the current directory? Should this be the root directory of the current crate?
    let mut path_with_graphblas_implementation = std::env::current_dir().unwrap();
    path_with_graphblas_implementation.push("graphblas_implementation");
    return path_with_graphblas_implementation;
}

fn path_with_suitesparse_graphblas_implementation() -> PathBuf {
    let mut path_with_suitesparse_graphblas_implementation = path_with_graphblas_implementation();
    path_with_suitesparse_graphblas_implementation.push("SuiteSparse_GraphBLAS");
    return path_with_suitesparse_graphblas_implementation;
}

fn path_with_graphblas_header_file() -> PathBuf {
    let mut path_with_graphblas_header_file = path_with_suitesparse_graphblas_implementation();
    path_with_graphblas_header_file.push("Include");
    path_with_graphblas_header_file.push("GraphBLAS.h");
    return path_with_graphblas_header_file;
}

fn path_with_graphblas_library() -> PathBuf {
    let mut path_with_graphblas_library = path_with_graphblas_implementation();
    path_with_graphblas_library.push("lib");
    return path_with_graphblas_library;
}

// DOC: to set a persistent environment variable in Ubuntu:
// sudoedit /etc/profile
// export KEY="value"
// restart VM (on WSL => wsl --shutdown)
fn path_with_openmp() -> PathBuf {
    match std::env::var("SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH") { 
        Ok(path) => return PathBuf::from(path),
        Err(error) => {
            match search_compiler_path() {
                Some(path) => {
                    println!("Unable to read environment variable SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH: {}", error);
                    println!("Automatically use default compiler path: {}", path.display());
                    return path
                },
                None => {
                    panic!("Unable to read environment variable SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH. For example, when using GCC on Ubuntu 22.04, please set it to \"/usr/lib/gcc/x86_64-linux-gnu/11\". {}", error)
                }
            }
            
        }
    }
}

fn search_compiler_path() -> Option<PathBuf> {
    if cfg!(target_os = "linux") {
        let mut matching_paths = Vec::<PathBuf>::new();
        match glob("/usr/bin/**/libgcc.so") {
            Ok(paths) => {
                for entry in paths {
                    match entry {
                        Ok(path) => matching_paths.push(path),
                        Err(error) => println!("{}", error)
                    };
                };
            },
            Err(error) => {
                println!("{}", error);
                println!("Unable to automatically find an installed GCC compiler under \"/usr/bin/\", please install GCC or set the SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH environment variable.");
                return None
            }
        }
        if matching_paths.len() > 0 {
            matching_paths.sort();
            return Some(matching_paths.last().unwrap().to_owned())
        } else {
            println!("Unable to automatically find an installed GCC compiler under \"/usr/bin/\", please install GCC or set the SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH environment variable.");
            return None
        }
    } else {
        println!("Automatically finding an installed C compiler not supported on this operating system, please the SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH environment variable.");
        return None
    }
}

// DOC: to set a persistent environment variable in Ubuntu:
// sudoedit /etc/profile
// export KEY="value"
// restart VM (on WSL => wsl --shutdown)
fn name_of_openmp_library() -> String {
    match std::env::var("SUITESPARSE_GRAPHBLAS_SYS_OPENMP_STATIC_LIBRARY_NAME") { 
        Ok(name) => return name,
        Err(error) => {
            println!("{}", error);
            println!("Unable read environment variable SUITESPARSE_GRAPHBLAS_SYS_OPENMP_STATIC_LIBRARY_NAME, set to default \"gomp\"");
            return search_openmp().unwrap();
            // panic!("Unable to read environment variable SUITESPARSE_GRAPHBLAS_SYS_OPENMP_STATIC_LIBRARY_NAME. For example, when using GCC, please set it to \"gomp\". {}", error)
        }
    }
}

// TODO: as soon as automatic searching is supported for non-linux operating systems, and other compilers than GCC,
// this function must be implemented with an actual search algorithm.
fn search_openmp() -> Option<String> {
    return Some(String::from("gomp"))
}

fn build_and_link_dependencies() {
    let cargo_build_directory = env::var_os("OUT_DIR").unwrap();
    let path_with_graphblas_header_file = path_with_graphblas_header_file();
    let path_with_graphblas_implementation = path_with_graphblas_implementation();
    let path_with_suitesparse_graphblas_implementation =
        path_with_suitesparse_graphblas_implementation();

    // SuiteSparse::GraphBLAS repository is too large to fit a crate on crates.io (repo exceeds maximum allowed size of 10MB)
    clone_and_checkout_repository(&path_with_graphblas_header_file, &path_with_suitesparse_graphblas_implementation);

    build_static_graphblas_implementation(&cargo_build_directory);

    // add directory containing libgomp.a to search path.
    // This directory should also contain libgraphblas.a, if it is not build from source or otherwise in the search path
    println!(
        "cargo:rustc-link-search=native={}",
        path_with_suitesparse_graphblas_implementation
            .to_str()
            .unwrap()
            .to_owned()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        path_with_openmp().display()
    );

    println!("cargo:rustc-link-lib=static=graphblas");
    println!("cargo:rustc-link-lib=static={}", name_of_openmp_library());

    declare_build_invalidation_conditions(
        &path_with_graphblas_implementation,
        &path_with_graphblas_header_file,
    );

    clean_build_artifacts(&cargo_build_directory)
}

fn clone_and_checkout_repository(path_with_graphblas_header_file: &PathBuf, path_with_suitesparse_graphblas_implementation: &PathBuf) {
    let graphblas_repo;
    match path_with_graphblas_header_file.try_exists() {
        Err(error) => {
            panic!(
                "Unable to check if path {} exists, got error: {}",
                path_with_graphblas_header_file.display(),
                error
            )
        }
        Ok(does_exist) => {
            if !does_exist {
                graphblas_repo = match Repository::clone(
                    "https://github.com/DrTimothyAldenDavis/GraphBLAS",
                    path_with_suitesparse_graphblas_implementation.clone(),
                ) {
                    Ok(repo) => repo,
                    Err(error) => panic!("Failed to clone graphblas repository: {}", error),
                };
            } else {
                // assume repo has been cloned before
                graphblas_repo = match Repository::open(
                    path_with_suitesparse_graphblas_implementation.clone(),
                ) {
                    Ok(repo) => repo,
                    Err(error) => {
                        panic!("failed to open SuiteSparse GraphBLAS repository: {}", error)
                    }
                };
            }
        }
    };

    let obj: Object = graphblas_repo
        .find_commit(Oid::from_str("97510b55fba589e6ea315fe433237633057e7048").unwrap())
        .unwrap()
        .into_object();
    graphblas_repo.checkout_tree(&obj, None).unwrap();
    graphblas_repo.set_head_detached(obj.id()).unwrap();
}

fn build_static_graphblas_implementation(cargo_build_directory: &OsString) {
    let _dst = cmake::Config::new("graphblas_implementation/SuiteSparse_GraphBLAS")
        .define("BUILD_GRB_STATIC_LIBRARY", "true")
        .define("CMAKE_INSTALL_LIBDIR", cargo_build_directory.clone())
        .define("CMAKE_INSTALL_INCLUDEDIR", cargo_build_directory.clone())
        .define("PROJECT_SOURCE_DIR", cargo_build_directory.clone()) // prevent modifying config files outside of the cargo output directory
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        cargo_build_directory.clone().to_str().unwrap().to_owned()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        path_with_graphblas_library().display()
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
        path_with_openmp()
            .clone()
            .join(format!("lib{}.a",name_of_openmp_library()))
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
            "FP_NAN".into(),
            "FP_INFINITE".into(),
            "FP_ZERO".into(),
            "FP_SUBNORMAL".into(),
            "FP_NORMAL".into(),
            // "IPPORT_RESERVED".into(),
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
        .expect("Unable to generate bindings to GraphBLAS library.");

    let mut bindings_target_path = path_with_graphblas_implementation.clone();
    bindings_target_path.push("suitesparse_graphblas_bindings.rs");
    bindings
        .write_to_file(bindings_target_path.to_str().unwrap().to_owned())
        .expect("Couldn't write bindings!");
}

fn clean_build_artifacts(cargo_build_directory: &OsString) {
    let cargo_build_directory = PathBuf::from(cargo_build_directory);
    let mut path_to_delete = cargo_build_directory.clone();
    path_to_delete.push("build");
    fs::remove_dir_all(path_to_delete).is_ok();

    let path_to_delete_files_in = cargo_build_directory;
    let mut path_to_keep = path_to_delete_files_in.clone();
    path_to_keep.push("libgraphblas.a");

    for path in fs::read_dir(path_to_delete_files_in).unwrap() {
        let path = path.unwrap();
        let path = path.path();

        if path.clone().into_os_string().into_string().unwrap()
            != path_to_keep.clone().into_os_string().into_string().unwrap()
        {
            fs::remove_file(path).is_ok();
        }
    }
}
