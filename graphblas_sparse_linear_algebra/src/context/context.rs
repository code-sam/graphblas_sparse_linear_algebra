#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::mem::MaybeUninit;
use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use once_cell::sync::Lazy;
use suitesparse_graphblas_sys::{
    GrB_BinaryOp, GrB_BinaryOp_error, GrB_Descriptor, GrB_Descriptor_error, GrB_IndexUnaryOp,
    GrB_IndexUnaryOp_error, GrB_Matrix, GrB_Matrix_error, GrB_Monoid, GrB_Monoid_error, GrB_Scalar,
    GrB_Scalar_error, GrB_Semiring, GrB_Semiring_error, GrB_Type, GrB_Type_error, GrB_UnaryOp,
    GrB_UnaryOp_error, GrB_Vector, GrB_Vector_error, GrB_finalize, GxB_SelectOp,
    GxB_SelectOp_error,
};

use crate::bindings_to_graphblas_implementation::{
    GrB_Info,
    GrB_Info_GrB_DIMENSION_MISMATCH,
    GrB_Info_GrB_DOMAIN_MISMATCH,
    GrB_Info_GrB_EMPTY_OBJECT,
    GrB_Info_GrB_INDEX_OUT_OF_BOUNDS,
    GrB_Info_GrB_INSUFFICIENT_SPACE,
    GrB_Info_GrB_INVALID_INDEX,
    GrB_Info_GrB_INVALID_OBJECT,
    GrB_Info_GrB_INVALID_VALUE,
    GrB_Info_GrB_NOT_IMPLEMENTED,
    GrB_Info_GrB_NO_VALUE,
    GrB_Info_GrB_NULL_POINTER,
    GrB_Info_GrB_OUTPUT_NOT_EMPTY,
    GrB_Info_GrB_OUT_OF_MEMORY,
    GrB_Info_GrB_PANIC,
    GrB_Info_GrB_SUCCESS,
    GrB_Info_GrB_UNINITIALIZED_OBJECT,
    GrB_Info_GxB_EXHAUSTED,
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
If this expectation is false, or given the current GraphBLAS C-specifcation version 2.0,
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

pub trait ContextTrait {
    fn context(&self) -> Arc<Context>;
    fn context_ref(&self) -> &Arc<Context>;
}

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
            let status = initialize(mode.to_owned(), number_of_ready_contexts)?;
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

    // TODO: check context is Ready
    pub fn call_without_detailed_error_information<F>(
        &self,
        mut function_to_call: F,
    ) -> Result<Status, SparseLinearAlgebraError>
    where
        F: FnMut() -> GrB_Info,
    {
        call_graphblas_implementation_without_detailed_error_information(function_to_call)
    }
}

fn initialize(
    mode: Mode,
    number_of_ready_contexts: MutexGuard<AtomicIsize>,
) -> Result<Status, SparseLinearAlgebraError> {
    // println!("Trying to initialize a context");
    // let _is_graphblas_busy = IS_GRAPHBLAS_BUSY.lock().unwrap();
    // println!("Got a lock to initialize a context! {:?}", is_graphblas_busy.load(Ordering::SeqCst));
    let status = unsafe {
        graphblas_result(GrB_init(mode.into()), || -> String {
            String::from("Something went wrong while initializing a GraphBLAS context")
        })?
    };
    number_of_ready_contexts.fetch_add(1, Ordering::SeqCst);
    // println!("Initialised a context");
    Ok(status)
}

pub trait CallGraphBlasContext<T> {
    fn call<F>(
        &self,
        function_to_call: F,
        reference_to_debug_info: &T,
    ) -> Result<Status, SparseLinearAlgebraError>
    where
        F: FnMut() -> GrB_Info;
}

#[derive(Debug, PartialEq)]
pub struct NotReady {}

#[derive(Debug, PartialEq)]
pub struct Ready {
    mode: Mode,
    // version: Version
}

impl Ready {
    fn call_without_detailed_error_information<F>(
        &self,
        mut function_to_call: F,
    ) -> Result<Status, SparseLinearAlgebraError>
    where
        F: FnMut() -> GrB_Info,
    {
        call_graphblas_implementation_without_detailed_error_information(function_to_call)
    }
}

fn call_graphblas_implementation_without_detailed_error_information<F>(
    mut function_to_call: F,
) -> Result<Status, SparseLinearAlgebraError>
where
    F: FnMut() -> GrB_Info,
{
    // thread::sleep(time::Duration::from_secs(2));
    // let _is_graphblas_busy = IS_GRAPHBLAS_BUSY.lock().unwrap();
    graphblas_result(function_to_call(), || -> String {
        String::from("Something went wrong while calling the GraphBLAS context.")
    })
}

impl Ready {
    fn finalize_context(&self) -> Result<Status, SparseLinearAlgebraError> {
        Ok(self.call_without_detailed_error_information(|| unsafe { GrB_finalize() })?)
    }
}

impl Drop for Ready {
    fn drop(&mut self) -> () {
        let number_of_ready_contexts = NUMBER_OF_READY_CONTEXTS.lock().unwrap();
        if number_of_ready_contexts.load(Ordering::SeqCst) == 0 {
            self.finalize_context().unwrap();
        }
    }
}

macro_rules! implement_CallGraphBlasContext {
    ($graphblas_type: ty, $error_retrieval_function: ident) => {
        paste::paste! {
            impl CallGraphBlasContext<$graphblas_type> for Context {
                // TODO: check Context state is Ready
                fn call<F>(
                    &self,
                    mut function_to_call: F,
                    reference_to_debug_info: &$graphblas_type,
                ) -> Result<Status, SparseLinearAlgebraError>
                where
                    F: FnMut() -> GrB_Info,
                {
                    let get_detailed_error_information =
                        [<generate_closure_to_retrieve_detailed_error_message_ $graphblas_type>](reference_to_debug_info);
                    // thread::sleep(time::Duration::from_secs(2));
                    // let _is_graphblas_busy = IS_GRAPHBLAS_BUSY.lock().unwrap();
                    graphblas_result(function_to_call(), get_detailed_error_information)
                }
            }

            fn [<generate_closure_to_retrieve_detailed_error_message_ $graphblas_type>]<'a>(
                struct_with_debugging_info: &'a $graphblas_type,
            ) -> impl Fn() -> String + 'a {
                return || -> String {
                    let mut graphblas_error_message: MaybeUninit<*const c_char> = MaybeUninit::uninit();
                    let graphblas_call_status;
                    unsafe {
                        // graphblas_call_status = suitesparse_graphblas_sys::GrB_error(
                        graphblas_call_status = $error_retrieval_function(
                            graphblas_error_message.as_mut_ptr(),
                            *struct_with_debugging_info,
                        );
                    }
                    let graphblas_error_message = unsafe { graphblas_error_message.assume_init() };
                    match graphblas_call_status {
                        GrB_Info_GrB_SUCCESS => {
                            let message;
                            unsafe {
                                message = CStr::from_ptr(graphblas_error_message).to_str();
                            }
                            match message {
                                Ok(message) => message.to_owned(),
                                Err(error) => format!("Something went wrong while calling the GraphBLAS implementation. Unable to parse detailed error message due to: {}", error)
                            }
                        }
                        _ => return String::from("Something went wrong while calling the GraphBLAS implementation. Unable to retrieve more detailed error information.")
                    }
                };
            }
        }
    };
}

implement_CallGraphBlasContext!(GrB_Type, GrB_Type_error);
implement_CallGraphBlasContext!(GrB_Scalar, GrB_Scalar_error);
implement_CallGraphBlasContext!(GrB_Vector, GrB_Vector_error);
implement_CallGraphBlasContext!(GrB_Matrix, GrB_Matrix_error);
implement_CallGraphBlasContext!(GrB_Descriptor, GrB_Descriptor_error);
implement_CallGraphBlasContext!(GrB_UnaryOp, GrB_UnaryOp_error);
implement_CallGraphBlasContext!(GrB_BinaryOp, GrB_BinaryOp_error);
implement_CallGraphBlasContext!(GrB_Semiring, GrB_Semiring_error);
implement_CallGraphBlasContext!(GrB_Monoid, GrB_Monoid_error);
implement_CallGraphBlasContext!(GrB_IndexUnaryOp, GrB_IndexUnaryOp_error);
implement_CallGraphBlasContext!(GxB_SelectOp, GxB_SelectOp_error);

fn graphblas_result<F>(
    grb_info: GrB_Info,
    get_detailed_error_information: F,
) -> Result<Status, SparseLinearAlgebraError>
where
    F: Fn() -> String,
{
    let status = Status::from(grb_info);
    match status {
        Status::Success => Ok(Status::Success),
        _ => Err(status.into_sparse_linear_algebra_error(get_detailed_error_information())),
    }
}

// TODO: consider the use of https://crates.io/crates/enum_primitive

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Success,
    NoValue,
    UnitializedObject,
    InvalidObject,
    NotImplemented,
    NullPointer,
    InvalidValue,
    InvalidIndex,
    DomainMismatch,
    DimensionMismatch,
    EmptyObject,
    OutputNotEmpty,
    OutOfMemory,
    InsufficientSpace,
    IndexOutOfBounds,
    IteratorExhausted,
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
            GrB_Info_GrB_NOT_IMPLEMENTED => Self::NotImplemented,
            GrB_Info_GrB_NULL_POINTER => Self::NullPointer,
            GrB_Info_GrB_INVALID_VALUE => Self::InvalidValue,
            GrB_Info_GrB_INVALID_INDEX => Self::InvalidIndex,
            GrB_Info_GrB_DOMAIN_MISMATCH => Self::DomainMismatch,
            GrB_Info_GrB_DIMENSION_MISMATCH => Self::DimensionMismatch,
            GrB_Info_GrB_EMPTY_OBJECT => Self::EmptyObject,
            GrB_Info_GrB_OUTPUT_NOT_EMPTY => Self::OutputNotEmpty,
            GrB_Info_GrB_OUT_OF_MEMORY => Self::OutOfMemory,
            GrB_Info_GrB_INSUFFICIENT_SPACE => Self::InsufficientSpace,
            GrB_Info_GrB_INDEX_OUT_OF_BOUNDS => Self::IndexOutOfBounds,
            GrB_Info_GxB_EXHAUSTED => Self::IteratorExhausted,
            GrB_Info_GrB_PANIC => Self::Panic,
            _ => Self::UnknownStatusType,
        }
    }
}

impl Status {
    fn into_sparse_linear_algebra_error(
        self,
        detailed_error_information: String,
    ) -> SparseLinearAlgebraError {
        match self {
            Status::Success => SystemError::new(
                SystemErrorType::CreateGraphBlasErrorOnSuccessValue,
                format!("Logic error, called into<GraphBlasError> for success status"),
                None,
            )
            .into(),
            Status::NoValue => {
                GraphBlasError::new(GraphBlasErrorType::NoValue, detailed_error_information).into()
            }
            Status::UnitializedObject => GraphBlasError::new(
                GraphBlasErrorType::UnitializedObject,
                detailed_error_information,
            )
            .into(),
            Status::InvalidObject => GraphBlasError::new(
                GraphBlasErrorType::InvalidObject,
                detailed_error_information,
            )
            .into(),
            Status::NotImplemented => GraphBlasError::new(
                GraphBlasErrorType::NotImplemented,
                detailed_error_information,
            )
            .into(),
            Status::NullPointer => {
                GraphBlasError::new(GraphBlasErrorType::NullPointer, detailed_error_information)
                    .into()
            }
            Status::InvalidValue => {
                GraphBlasError::new(GraphBlasErrorType::InvalidValue, detailed_error_information)
                    .into()
            }
            Status::InvalidIndex => {
                GraphBlasError::new(GraphBlasErrorType::InvalidIndex, detailed_error_information)
                    .into()
            }
            Status::DomainMismatch => GraphBlasError::new(
                GraphBlasErrorType::DomainMismatch,
                detailed_error_information,
            )
            .into(),
            Status::DimensionMismatch => GraphBlasError::new(
                GraphBlasErrorType::DimensionMismatch,
                detailed_error_information,
            )
            .into(),
            Status::EmptyObject => {
                GraphBlasError::new(GraphBlasErrorType::EmptyObject, detailed_error_information)
                    .into()
            }
            Status::OutputNotEmpty => GraphBlasError::new(
                GraphBlasErrorType::OutputNotEmpty,
                detailed_error_information,
            )
            .into(),
            Status::OutOfMemory => {
                GraphBlasError::new(GraphBlasErrorType::OutOfMemory, detailed_error_information)
                    .into()
            }
            Status::InsufficientSpace => GraphBlasError::new(
                GraphBlasErrorType::InsufficientSpace,
                detailed_error_information,
            )
            .into(),
            Status::IndexOutOfBounds => GraphBlasError::new(
                GraphBlasErrorType::IndexOutOfBounds,
                detailed_error_information,
            )
            .into(),
            Status::IteratorExhausted => GraphBlasError::new(
                GraphBlasErrorType::IteratorExhausted,
                detailed_error_information,
            )
            .into(),
            Status::Panic => {
                GraphBlasError::new(GraphBlasErrorType::Panic, detailed_error_information).into()
            }
            Status::UnknownStatusType => SystemError::new(
                SystemErrorType::UnsupportedGraphBlasErrorValue,
                String::from("Something went wrong while calling the GrapBLAS implementation"),
                None,
            )
            .into(),
        }
    }
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
