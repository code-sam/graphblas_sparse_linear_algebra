use std::io::Read;
use std::{collections::HashSet, fs::File};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;

use git2::{Object, Oid, Repository, Revwalk};

extern crate bindgen;
extern crate cmake;

// NOTE: when updating the version, make sure to delete any existing clones
const GIT_COMMIT: &str = "5a3c6a81683de12c33eb3225ccb861f0531c1376";

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

// fn path_with_graphblas_jit_cache() -> PathBuf {
//     let mut path_with_graphblas_jit_cache = path_with_suitesparse_graphblas_implementation();
//     return path_with_graphblas_jit_cache;
// }

fn path_with_graphblas_cmakelists_file() -> PathBuf {
    let mut path_with_graphblas_cmakelists_file = path_with_suitesparse_graphblas_implementation();
    path_with_graphblas_cmakelists_file.push("CMakeLists.txt");
    return path_with_graphblas_cmakelists_file;
}

// DOC: to set a persistent environment variable in Ubuntu:
// sudoedit /etc/profile
// export KEY="value"
// restart VM (on WSL => wsl --shutdown)
fn path_with_openmp() -> PathBuf {
    match std::env::var("SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH") {
        Ok(path) => return PathBuf::from(path),
        Err(error) => match search_compiler_path() {
            Some(path) => {
                println!("Unable to read environment variable SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH: {}", error);
                println!(
                    "Automatically use default compiler path: {}",
                    path.display()
                );
                return path;
            }
            None => {
                panic!("Unable to read environment variable SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH. For example, when using GCC on Ubuntu 22.04, please set it to \"/usr/lib/gcc/x86_64-linux-gnu/11\". {}", error)
            }
        },
    }
}

fn search_compiler_path() -> Option<PathBuf> {
    if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        let mut path = path_with_graphblas_implementation();
        path.push("OpenMP");
        path.push("linux");
        Some(path)
    } else {
        println!("Unable to select a default OpenMP archive for this operating system and architecture, please the SUITESPARSE_GRAPHBLAS_SYS_COMPILER_PATH environment variable.");
        return None;
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
    return Some(String::from("gomp"));
}

fn build_and_link_dependencies() {
    let cargo_build_directory = env::var_os("OUT_DIR").unwrap();
    let path_with_graphblas_header_file = path_with_graphblas_header_file();
    let path_with_graphblas_implementation = path_with_graphblas_implementation();
    let path_with_suitesparse_graphblas_implementation =
        path_with_suitesparse_graphblas_implementation();
    let path_with_graphblas_cmakelists_file = path_with_graphblas_cmakelists_file();

    // SuiteSparse::GraphBLAS repository is too large to fit a crate on crates.io (repo exceeds maximum allowed size of 10MB)
    clone_and_checkout_repository(
        &path_with_graphblas_header_file,
        &path_with_suitesparse_graphblas_implementation,
    );

    // Modify the CMakeLists.txt file to force NSTATIC=0. This is a hack, somehow, the NSTATIC flag is not passed by cmake.
    customize_build_instructions(&path_with_graphblas_cmakelists_file);

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

fn clone_and_checkout_repository(
    path_with_graphblas_header_file: &PathBuf,
    path_with_suitesparse_graphblas_implementation: &PathBuf,
) {
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
                    path_with_suitesparse_graphblas_implementation.to_owned(),
                ) {
                    Ok(repo) => repo,
                    Err(error) => panic!("Failed to clone graphblas repository: {}", error),
                };
            } else {
                // assume repo has been cloned before
                graphblas_repo = match Repository::open(
                    path_with_suitesparse_graphblas_implementation.to_owned(),
                ) {
                    Ok(repo) => {
                        fast_forward(&repo);
                        repo
                    }
                    Err(error) => {
                        panic!("failed to open SuiteSparse GraphBLAS repository: {}", error)
                    }
                };
            }
        }
    };

    // Use for debugging purposes, i.e. find available commit number
    // print_commits(&graphblas_repo);

    let obj: Object = graphblas_repo
        .find_commit(Oid::from_str(GIT_COMMIT).unwrap())
        .unwrap()
        .into_object();
    graphblas_repo.checkout_tree(&obj, None).unwrap();
    graphblas_repo.set_head_detached(obj.id()).unwrap();
}

fn customize_build_instructions(path_with_graphblas_cmakelists_file: &PathBuf) {
    let mut file = File::open(path_with_graphblas_cmakelists_file).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let customized_contents = contents.replace("set ( NSTATIC_DEFAULT_ON true )", "set ( NSTATIC_DEFAULT_ON false )");
    fs::write(path_with_graphblas_cmakelists_file, customized_contents);
}

// Use for debugging purposes, i.e. find available commit number
fn print_commits(repo: &Repository) {
    // Create a Revwalk object
    let mut revwalk = repo.revwalk().unwrap();

    // Push the range of commits you want to walk through
    // Here, we're pushing all commits reachable from HEAD
    revwalk.push_head().unwrap();

    // Iterate over the commits
    for commit_id in revwalk {
        match commit_id {
            Ok(id) => {
                let commit = repo.find_commit(id).unwrap();
                println!("Commit: {}", commit.id());
                println!(
                    "Message: {}",
                    commit.message().unwrap_or("No commit message")
                );
            }
            Err(e) => println!("Error: {}", e),
        }
    }
}

fn fast_forward(repo: &Repository) {
    repo.find_remote("origin")
        .unwrap()
        .fetch(&["stable"], None, None)
        .unwrap();

    let fetch_head = repo.find_reference("FETCH_HEAD").unwrap();
    let fetch_commit = repo.reference_to_annotated_commit(&fetch_head).unwrap();
    let analysis = repo.merge_analysis(&[&fetch_commit]).unwrap();
    if analysis.0.is_up_to_date() {
        return;
    } else if analysis.0.is_fast_forward() {
        let refname = format!("refs/heads/{}", "stable");
        let mut reference = repo.find_reference(&refname).unwrap();
        reference
            .set_target(fetch_commit.id(), "Fast-Forward")
            .unwrap();
        repo.set_head(&refname).unwrap();
        repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))
            .unwrap()
    } else {
        panic!("Fast-forward only!")
    }
}

fn build_static_graphblas_implementation(cargo_build_directory: &OsString) {
    let mut build_configuration =
        cmake::Config::new("graphblas_implementation/SuiteSparse_GraphBLAS");

    build_configuration
        .define("NSTATIC", "false")
        .define("CMAKE_INSTALL_LIBDIR", cargo_build_directory.to_owned())
        .define("CMAKE_INSTALL_INCLUDEDIR", cargo_build_directory.to_owned())
        .define("PROJECT_SOURCE_DIR", cargo_build_directory.to_owned());

    if !cfg!(feature = "build-standard-kernels") {
        build_configuration.define("COMPACT", "true");
    }

    if cfg!(feature = "disable-just-in-time-compiler") {
        build_configuration.define("NJIT", "true");
    }

    let _dst = build_configuration.build();

    println!(
        "cargo:rustc-link-search=native={}",
        cargo_build_directory
            .to_owned()
            .to_str()
            .unwrap()
            .to_owned()
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
            .to_owned()
            .join("wrapper.h")
            .to_str()
            .unwrap()
            .to_owned()
    );
    println!(
        "cargo:rerun-if-changed = {}",
        path_with_graphblas_header_file
            .to_owned()
            .to_str()
            .unwrap()
            .to_owned()
    );
    println!(
        "cargo:rerun-if-changed = {}",
        path_with_openmp()
            .to_owned()
            .join(format!("lib{}.a", name_of_openmp_library()))
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
                .to_owned()
                .to_str()
                .unwrap()
                .to_owned(),
        )
        .parse_callbacks(Box::new(ignored_macros))
        .generate()
        .expect("Unable to generate bindings to GraphBLAS library.");

    let mut bindings_target_path = path_with_graphblas_implementation.to_owned();
    bindings_target_path.push("suitesparse_graphblas_bindings.rs");
    bindings
        .write_to_file(bindings_target_path.to_str().unwrap().to_owned())
        .expect("Couldn't write bindings!");
}

fn clean_build_artifacts(cargo_build_directory: &OsString) {
    let cargo_build_directory = PathBuf::from(cargo_build_directory);
    let mut path_to_delete = cargo_build_directory.to_owned();
    path_to_delete.push("build");
    fs::remove_dir_all(path_to_delete).is_ok();

    let path_to_delete_files_in = cargo_build_directory;
    let mut path_to_keep = path_to_delete_files_in.to_owned();
    path_to_keep.push("libgraphblas.a");

    for path in fs::read_dir(path_to_delete_files_in).unwrap() {
        let path = path.unwrap();
        let path = path.path();

        if path.to_owned().into_os_string().into_string().unwrap()
            != path_to_keep
                .to_owned()
                .into_os_string()
                .into_string()
                .unwrap()
        {
            fs::remove_file(path).is_ok();
        }
    }
}
