use std::{ptr, sync::Arc};

use suitesparse_graphblas_sys::{GrB_Matrix, GrB_Vector};

use crate::context::{Context, GetContext};

pub trait MatrixMask: GetContext {
    unsafe fn graphblas_matrix_ptr(&self) -> GrB_Matrix;
}

pub trait VectorMask: GetContext {
    unsafe fn graphblas_vector_ptr(&self) -> GrB_Vector;
}

#[derive(Debug, Clone)]
pub struct SelectEntireMatrix {
    context: Arc<Context>,
}

impl SelectEntireMatrix {
    pub fn new(context: Arc<Context>) -> Self {
        SelectEntireMatrix { context }
    }
}

impl MatrixMask for SelectEntireMatrix {
    unsafe fn graphblas_matrix_ptr(&self) -> GrB_Matrix {
        ptr::null_mut()
    }
}

impl GetContext for SelectEntireMatrix {
    fn context(&self) -> Arc<Context> {
        self.context.clone()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

#[derive(Debug, Clone)]
pub struct SelectEntireVector {
    context: Arc<Context>,
}

impl SelectEntireVector {
    pub fn new(context: Arc<Context>) -> Self {
        SelectEntireVector { context }
    }
}

impl VectorMask for SelectEntireVector {
    unsafe fn graphblas_vector_ptr(&self) -> GrB_Vector {
        ptr::null_mut()
    }
}

impl GetContext for SelectEntireVector {
    fn context(&self) -> Arc<Context> {
        self.context.clone()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}
