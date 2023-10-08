use std::{ptr, sync::Arc};

use suitesparse_graphblas_sys::{GrB_Matrix, GrB_Vector};

use crate::context::{Context, GetContext};

pub trait MatrixMask {
    unsafe fn graphblas_matrix(&self) -> GrB_Matrix;
}

pub trait VectorMask {
    unsafe fn graphblas_vector(&self) -> GrB_Vector;
}

#[derive(Debug, Clone)]
pub struct SelectEntireMatrix {
    context: Arc<Context>,
}

impl SelectEntireMatrix {
    pub fn new(context: &Arc<Context>) -> Self {
        SelectEntireMatrix {
            context: context.to_owned(),
        }
    }
}

impl MatrixMask for SelectEntireMatrix {
    unsafe fn graphblas_matrix(&self) -> GrB_Matrix {
        ptr::null_mut()
    }
}

impl GetContext for SelectEntireMatrix {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
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
    pub fn new(context: &Arc<Context>) -> Self {
        SelectEntireVector {
            context: context.to_owned(),
        }
    }
}

impl VectorMask for SelectEntireVector {
    unsafe fn graphblas_vector(&self) -> GrB_Vector {
        ptr::null_mut()
    }
}

impl GetContext for SelectEntireVector {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}
