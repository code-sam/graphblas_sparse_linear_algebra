#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use once_cell::sync::Lazy;

use super::super::bindings_to_graphblas_implementation::{
    GrB_Info,
    GrB_Info_GrB_DIMENSION_MISMATCH,
    GrB_Info_GrB_DOMAIN_MISMATCH,
    GrB_Info_GrB_INDEX_OUT_OF_BOUNDS,
    GrB_Info_GrB_INSUFFICIENT_SPACE,
    GrB_Info_GrB_INVALID_INDEX,
    GrB_Info_GrB_INVALID_OBJECT,
    GrB_Info_GrB_INVALID_VALUE,
    GrB_Info_GrB_NO_VALUE,
    GrB_Info_GrB_NULL_POINTER,
    GrB_Info_GrB_OUTPUT_NOT_EMPTY,
    GrB_Info_GrB_OUT_OF_MEMORY,
    GrB_Info_GrB_PANIC,
    GrB_Info_GrB_SUCCESS,
    GrB_Info_GrB_UNINITIALIZED_OBJECT,
    GrB_Mode,
    GrB_Mode_GrB_BLOCKING,
    GrB_Mode_GrB_NONBLOCKING,
    // GrB_error,
    GrB_init,
};

use crate::error::SparseLinearAlgebraError;
use crate::error::{GraphBlasError, GraphBlasErrorType};
use crate::error::{SystemError, SystemErrorType};

/*
TO REVIEW: The GraphBLAS context can only be initialized once per process (i.e. not per thread)
Also, after calling GrB_finalize(), the process must be restarted before GrB_init() can be called again.
These limtations are not compatible with the context concept implemented in this module.

This module assumes that multiple contexts can be started, closed, and re-started per thread.
It is expected that a future-version of GraphBLAS will support this concept.
If this expectation is false, or given the current GraphBLAS C-specifcation version 1.3,
the context concet implemented in this module is not useful.
*/

// lazy_static! {
//     static ref NUMBER_OF_READY_CONTEXTS: Mutex<AtomicIsize> = Mutex::new(AtomicIsize::new(0));
// }

// lazy_static! {
//     static ref IS_GRAPHBLAS_BUSY: Mutex<AtomicBool> = Mutex::new(AtomicBool::new(false));
// }

// static NUMBER_OF_READY_CONTEXTS: Arc<Mutex<AtomicIsize>> = Arc::new(Mutex::new(AtomicIsize::new(0)));
// static IS_GRAPHBLAS_BUSY: Arc<Mutex<AtomicBool>> = Arc::new(Mutex::new(AtomicBool::new(false)));

static NUMBER_OF_READY_CONTEXTS: Lazy<Mutex<AtomicIsize>> =
    Lazy::new(|| Mutex::new(AtomicIsize::new(0)));
static IS_GRAPHBLAS_BUSY: Lazy<Mutex<AtomicBool>> =
    Lazy::new(|| Mutex::new(AtomicBool::new(false)));

// fn is_context_initialized(number_of_ready_contexts: MutexGuard<AtomicUsize>) -> bool {
//     number_of_ready_contexts.load(Ordering::SeqCst) > 0
// }

#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    Blocking,
    NonBlocking,
}

impl From<GrB_Mode> for Mode {
    fn from(mode: GrB_Mode) -> Self {
        match mode {
            GrB_Mode_GrB_BLOCKING => Self::Blocking,
            GrB_Mode_GrB_NONBLOCKING => Self::NonBlocking,
            _ => panic!("Context mode not supported: {}", mode),
        }
    }
}

impl Into<GrB_Mode> for Mode {
    fn into(self) -> GrB_Mode {
        match self {
            Self::Blocking => GrB_Mode_GrB_BLOCKING,
            Self::NonBlocking => GrB_Mode_GrB_NONBLOCKING,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Context {
    Ready(Ready),
    NotReady(NotReady),
}

impl Context {
    fn new() -> Self {
        Context::NotReady(NotReady {})
    }

    pub fn init_ready(_mode: Mode) -> Result<Arc<Self>, SparseLinearAlgebraError> {
        let mut context = Context::new();
        context.start(Mode::NonBlocking)?;
        Ok(Arc::new(context))
    }

    fn start(&mut self, mode: Mode) -> Result<Status, SparseLinearAlgebraError> {
        let number_of_ready_contexts = NUMBER_OF_READY_CONTEXTS.lock().unwrap();
        // println!("number_of_ready_contexts before starting: {:?}",number_of_ready_contexts.load(Ordering::SeqCst));
        if number_of_ready_contexts.load(Ordering::SeqCst) == 0 {
            let status = initialize(mode.clone(), number_of_ready_contexts)?;
            *self = Context::Ready(Ready { mode });
            Ok(status)
        } else {
            number_of_ready_contexts.fetch_add(1, Ordering::SeqCst);
            *self = Context::Ready(Ready { mode });
            Ok(Status::Success)
        }
    }

    // TODO: make this safe to use. At the moment, the graphblas context is dropped automatically after all contexts have been dropped.
    // pub fn stop(&mut self) -> () {
    //     match &*self {
    //         Context::Ready(ready) => {
    //             *self = Context::NotReady(NotReady {});
    //         }
    //         Context::NotReady(_) => (),
    //     }
    // }

    // TODO: check context state
    pub fn call<F>(&self, mut function_to_call: F) -> Result<Status, SparseLinearAlgebraError>
    where
        F: FnMut() -> GrB_Info,
    {
        // thread::sleep(time::Duration::from_secs(2));
        let _is_graphblas_busy = IS_GRAPHBLAS_BUSY.lock().unwrap();
        graphblas_result(function_to_call())
    }
}

fn initialize(
    mode: Mode,
    number_of_ready_contexts: MutexGuard<AtomicIsize>,
) -> Result<Status, SparseLinearAlgebraError> {
    // println!("Trying to initialize a context");
    let _is_graphblas_busy = IS_GRAPHBLAS_BUSY.lock().unwrap();
    // println!("Got a lock to initialize a context! {:?}", is_graphblas_busy.load(Ordering::SeqCst));
    let status = unsafe { graphblas_result(GrB_init(mode.into()))? };
    number_of_ready_contexts.fetch_add(1, Ordering::SeqCst);
    // println!("Initialised a context");
    Ok(status)
}

#[derive(Debug, PartialEq)]
pub struct NotReady {}

#[derive(Debug, PartialEq)]
pub struct Ready {
    mode: Mode,
    // version: Version
}

// pub trait Call {
//     fn call<F>(&self, function_to_call: F) -> Result<Status, GraphBlasError>
//     where
//         F: FnMut() -> GrB_Info;
// }

// impl Call for NotReady {
//     fn call<F>(&self, function_to_call: F) -> Result<Status, GraphBlasError>
//     where
//         F: FnMut() -> GrB_Info,
//     {
//         Err(GraphBlasError::new(GraphBlasErrorType::UninitialisedContext, String::from("Cannot call GraphBLAS before it's context is initialised")))
//     }
// }

impl Ready {
    fn call<F>(&self, mut function_to_call: F) -> Result<Status, SparseLinearAlgebraError>
    where
        F: FnMut() -> GrB_Info,
    {
        // thread::sleep(time::Duration::from_secs(2));
        let _is_graphblas_busy = IS_GRAPHBLAS_BUSY.lock().unwrap();
        graphblas_result(function_to_call())
    }
}

impl Ready {
    fn finalize_context(&self) -> () {
        // match self.call(|| unsafe { GrB_finalize() }) {
        //     Ok(_) => {
        //         // thread::sleep(time::Duration::from_secs(2)); // allocate some time for Graphblas to shut down
        //         ()
        //     },
        //     Err(error) => panic!(error),
        // }
    }
}

impl Drop for Ready {
    fn drop(&mut self) -> () {
        let number_of_ready_contexts = NUMBER_OF_READY_CONTEXTS.lock().unwrap();
        if number_of_ready_contexts.load(Ordering::SeqCst) == 0 {
            self.finalize_context();
        }
    }
}

fn graphblas_result(grb_info: GrB_Info) -> Result<Status, SparseLinearAlgebraError> {
    let status = Status::from(grb_info);
    match status {
        Status::Success => Ok(Status::Success),
        _ => Err(status.into()),
    }
}

// TODO: consider the use of https://crates.io/crates/enum_primitive

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Success,
    NoValue,
    UnitializedObject,
    InvalidObject,
    NullPointer,
    InvalidValue,
    InvalidIndex,
    DomainMismatch,
    DimensionMismatch,
    OutputNotEmpty,
    OutOfMemory,
    InsufficientSpace,
    IndexOutOfBounds,
    Panic,
    UnknownStatusType,
}

impl From<GrB_Info> for Status {
    fn from(status: GrB_Info) -> Self {
        match status {
            GrB_Info_GrB_SUCCESS => Self::Success,
            GrB_Info_GrB_NO_VALUE => Self::NoValue,
            GrB_Info_GrB_UNINITIALIZED_OBJECT => Self::UnitializedObject,
            GrB_Info_GrB_INVALID_OBJECT => Self::InvalidObject,
            GrB_Info_GrB_NULL_POINTER => Self::NullPointer,
            GrB_Info_GrB_INVALID_VALUE => Self::InvalidValue,
            GrB_Info_GrB_INVALID_INDEX => Self::InvalidIndex,
            GrB_Info_GrB_DOMAIN_MISMATCH => Self::DomainMismatch,
            GrB_Info_GrB_DIMENSION_MISMATCH => Self::DimensionMismatch,
            GrB_Info_GrB_OUTPUT_NOT_EMPTY => Self::OutputNotEmpty,
            GrB_Info_GrB_OUT_OF_MEMORY => Self::OutOfMemory,
            GrB_Info_GrB_INSUFFICIENT_SPACE => Self::InsufficientSpace,
            GrB_Info_GrB_INDEX_OUT_OF_BOUNDS => Self::IndexOutOfBounds,
            GrB_Info_GrB_PANIC => Self::Panic,
            _ => Self::UnknownStatusType,
        }
    }
}

impl Into<SparseLinearAlgebraError> for Status {
    fn into(self) -> SparseLinearAlgebraError {
        match self {
            Status::Success => SystemError::new(
                SystemErrorType::CreateGraphBlasErrorOnSuccessValue,
                format!("Logic error, called into<GraphBlasError> for success status"),
                None,
            )
            .into(),
            Status::NoValue => {
                GraphBlasError::new(GraphBlasErrorType::NoValue, get_error_context()).into()
            }
            Status::UnitializedObject => {
                GraphBlasError::new(GraphBlasErrorType::UnitializedObject, get_error_context())
                    .into()
            }
            Status::InvalidObject => {
                GraphBlasError::new(GraphBlasErrorType::InvalidObject, get_error_context()).into()
            }
            Status::NullPointer => {
                GraphBlasError::new(GraphBlasErrorType::NullPointer, get_error_context()).into()
            }
            Status::InvalidValue => {
                GraphBlasError::new(GraphBlasErrorType::InvalidValue, get_error_context()).into()
            }
            Status::InvalidIndex => {
                GraphBlasError::new(GraphBlasErrorType::InvalidIndex, get_error_context()).into()
            }
            Status::DomainMismatch => {
                GraphBlasError::new(GraphBlasErrorType::DomainMismatch, get_error_context()).into()
            }
            Status::DimensionMismatch => {
                GraphBlasError::new(GraphBlasErrorType::DimensionMismatch, get_error_context())
                    .into()
            }
            Status::OutputNotEmpty => {
                GraphBlasError::new(GraphBlasErrorType::OutputNotEmpty, get_error_context()).into()
            }
            Status::OutOfMemory => {
                GraphBlasError::new(GraphBlasErrorType::OutOfMemory, get_error_context()).into()
            }
            Status::InsufficientSpace => {
                GraphBlasError::new(GraphBlasErrorType::InsufficientSpace, get_error_context())
                    .into()
            }
            Status::IndexOutOfBounds => {
                GraphBlasError::new(GraphBlasErrorType::IndexOutOfBounds, get_error_context())
                    .into()
            }
            Status::Panic => {
                GraphBlasError::new(GraphBlasErrorType::Panic, get_error_context()).into()
            }
            Status::UnknownStatusType => SystemError::new(
                SystemErrorType::UnsupportedGraphBlasErrorValue,
                get_error_context(),
                None,
            )
            .into(),
        }
    }
}

fn get_error_context() -> String {
    String::from("Something went wrong while calling GraphBLAS")
    // unsafe { CStr::from_ptr(GrB_error()).to_string_lossy().into_owned() }
}

// TODO: review why this FFI was not generated by bindgen
extern "C" {
    fn GrB_error() -> *const c_char;
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn init_graphblas() {
    //     let info = unsafe { GrB_init(GrB_Mode_GrB_NONBLOCKING) };
    //     let status = Status::from(info);
    //     println!("GrB_init status={:?}", status);
    //     assert_eq!(status, Status::Success);
    // }
    // #[test]
    // fn init_graphblas_2() {
    //     let info = unsafe { GrB_init(GrB_Mode_GrB_NONBLOCKING) };
    //     let status = Status::from(info);
    //     println!("GrB_init status={:?}", status);
    //     assert_eq!(status, Status::Success);
    // }

    #[test]
    fn start_and_drop_context() {
        let mut context = Context::new();
        context.start(Mode::NonBlocking).unwrap();

        // assert_eq!(
        //     context,
        //     Context::Ready(Ready {
        //         mode: Mode::NonBlocking
        //     })
        // );

        // // the Ready instance will get dropped and (falsely) substract 1 to the number of active contexts
        // // To compensate this test-specific error, manually increase the context count
        // let number_of_ready_contexts = NUMBER_OF_READY_CONTEXTS.lock().unwrap();
        // number_of_ready_contexts.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn start_and_drop_context_2() {
        let mut context = Context::new();
        context.start(Mode::NonBlocking).unwrap();
        // let mut context = Context::init_ready(Mode::NonBlocking).unwrap();

        // assert_eq!(
        //     context,
        //     Context::Ready(Ready {
        //         mode: Mode::NonBlocking
        //     })
        // );
        // // the Ready instance will get dropped and (falsely) substract 1 to the number of active contexts
        // // To compensate this test-specific error, manually increase the context count
        // let number_of_ready_contexts = NUMBER_OF_READY_CONTEXTS.lock().unwrap();
        // number_of_ready_contexts.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn start_and_drop_context_3() {
        let _context = Context::init_ready(Mode::NonBlocking).unwrap();

        // assert_eq!(
        //     context,
        //     Arc::new(Context::Ready(Ready {
        //         mode: Mode::NonBlocking
        //     }))
        // );

        // // the Ready instance will get dropped and (falsely) substract 1 to the number of active contexts
        // // To compensate this test-specific error, manually increase the context count
        // let number_of_ready_contexts = NUMBER_OF_READY_CONTEXTS.lock().unwrap();
        // number_of_ready_contexts.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn start_and_drop_context_4() {
        let _context = Context::init_ready(Mode::NonBlocking).unwrap();

        // assert_eq!(
        //     context,
        //     Arc::new(Context::Ready(Ready {
        //         mode: Mode::NonBlocking
        //     }))
        // );

        // // the Ready instance will get dropped and (falsely) substract 1 to the number of active contexts
        // // To compensate this test-specific error, manually increase the context count
        // let number_of_ready_contexts = NUMBER_OF_READY_CONTEXTS.lock().unwrap();
        // number_of_ready_contexts.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn start_and_drop_context_5() {
        let _context = Context::init_ready(Mode::NonBlocking).unwrap();

        // assert_eq!(
        //     context,
        //     Arc::new(Context::Ready(Ready {
        //         mode: Mode::NonBlocking
        //     }))
        // );

        // // the Ready instance will get dropped and (falsely) substract 1 to the number of active contexts
        // // To compensate this test-specific error, manually increase the context count
        // let number_of_ready_contexts = NUMBER_OF_READY_CONTEXTS.lock().unwrap();
        // number_of_ready_contexts.fetch_add(1, Ordering::SeqCst);
    }
}
