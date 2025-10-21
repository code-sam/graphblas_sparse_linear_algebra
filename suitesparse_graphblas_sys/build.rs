use std::collections::HashSet;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use git2::{Object, Oid, Repository};
use regex::Regex;

extern crate bindgen;
extern crate cmake;

// NOTE: when updating the version, make sure to delete any existing clones
const GIT_COMMIT: &str = "50cca249482611a47b6bf29c34b08bc8d7fc4644";

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

    // SuiteSparse::GraphBLAS repository is too large to fit a crate on crates.io (repo exceeds maximum allowed size of 10MB)
    clone_and_checkout_repository(
        &path_with_graphblas_header_file,
        &path_with_suitesparse_graphblas_implementation,
    );

    patch_gb_zstd_header(&path_with_suitesparse_graphblas_implementation);
    patch_zstd_iserror_symbols(&path_with_suitesparse_graphblas_implementation);

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
                        // During packaging, the source code must be immutable.
                        // Update the source code only during testing, this may be necessary if the SuiteSparse GraphBLAS
                        //  was already cloned before, but became outdated after updating to a new version.
                        // The update of the source code will now be performed automatically if the tests are run.
                        if cfg!(debug_assertions) {
                            fast_forward(&repo);
                        }
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
        .define("BUILD_SHARED_LIBS", "true")
        .define("GRAPHBLAS_BUILD_STATIC_LIBS", "true")
        .define("CMAKE_INSTALL_LIBDIR", cargo_build_directory.to_owned())
        .define("CMAKE_INSTALL_INCLUDEDIR", cargo_build_directory.to_owned())
        .define("PROJECT_SOURCE_DIR", cargo_build_directory.to_owned());

    if !cfg!(feature = "build-standard-kernels") {
        build_configuration.define("GRAPHBLAS_COMPACT", "true");
    }

    if cfg!(feature = "disable-just-in-time-compiler") {
        build_configuration.define("GRAPHBLAS_USE_JIT", "false");
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

fn patch_gb_zstd_header(graphblas_implementation_path: &Path) {
    let mut header_to_patch = PathBuf::from(graphblas_implementation_path);
    header_to_patch.push(Path::new(
        "Source/zstd_wrapper/GB_zstd.h",
    ));

    if !header_to_patch.exists() {
        println!(
            "cargo:warning={:?} not found; skipping patch",
            header_to_patch
        );
        return;
    }

    let content = fs::read_to_string(&header_to_patch).expect("Failed to read GB_zstd.h");

    // Only patch once (idempotent
    if content.contains("GB_FSE_isError") {
        println!("cargo:rerun-if-changed={}", header_to_patch.display());
        return;
    }

    let content = fs::read_to_string(&header_to_patch).expect("Failed to read GB_zstd.h");

    let patch = r#"
// Added automatically by build.rs to avoid zstd symbol collisions
#define FSE_isError  GB_FSE_isError
#define HUF_isError  GB_HUF_isError
"#;

    // Insert before final #endif
    let patched = if let Some(index) = content.rfind("#endif") {
        let (head, tail) = content.split_at(index);
        format!("{head}{patch}\n{tail}")
    } else {
        // fallback: append at the end
        format!("{content}\n{patch}")
    };

    fs::write(&header_to_patch, patched)
        .expect(&format!("Failed to patch {}", header_to_patch.display()));
    println!("cargo:warning=Patched GB_zstd.h to add FSE_isError/HUF_isError renames");
}

fn patch_zstd_iserror_symbols(graphblas_implementation_path: &Path) {
    let zstd_dir = PathBuf::from(graphblas_implementation_path)
        .join("zstd/zstd_subset");

    if !zstd_dir.exists() {
        println!(
            "cargo:warning=Zstd subset directory not found at {}; skipping patch",
            zstd_dir.display()
        );
        return;
    }

    println!(
        "cargo:warning=Patching Zstd subset recursively in {}",
        zstd_dir.display()
    );

    // Walk recursively
    fn walk_and_patch(dir: &Path) {
        for entry in fs::read_dir(dir).expect("Failed to read directory") {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();

            if path.is_dir() {
                walk_and_patch(&path);
            } else if path
                .extension()
                .map(|e| e == "c" || e == "h")
                .unwrap_or(false)
            {
                patch_file(&path);
            }
        }
    }

    // The actual patch logic
    fn patch_file(path: &Path) {
        let Ok(content) = fs::read_to_string(path) else {
            return;
        };

        let re = Regex::new(r"\b(ZSTD_isError)\b").unwrap();
        if !re.is_match(&content) {
            return; // nothing to do
        }

        let replaced = re.replace_all(&content, |caps: &regex::Captures| {
            match &caps[1] {
                "ZSTD_isError" => "GB_ZSTD_isError",
                _ => &caps[1],
            }
            .to_string()
        });

        if let Err(e) = fs::write(path, replaced.as_bytes()) {
            println!("cargo:warning=Failed to write {}: {}", path.display(), e);
        } else {
            println!("cargo:warning=Patched {}", path.display());
        }
    }

    walk_and_patch(&zstd_dir);
    println!("cargo:warning=Finished patching Zstd subset symbols.");
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
    let _ = fs::remove_dir_all(path_to_delete).is_ok();

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
            let _ = fs::remove_file(path).is_ok();
        }
    }
}
