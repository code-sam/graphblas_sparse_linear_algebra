use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use suitesparse_graphblas_sys::GrB_Matrix;

use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::context::{Context, GetContext};

pub struct SparseMatrixView<'a, T>
where
    T: GetGraphblasSparseMatrix + GetContext,
{
    raw_pointer: *const T,
    lifetime_marker: PhantomData<&'a T>,
}

impl<'a, T> SparseMatrixView<'a, T>
where
    T: GetGraphblasSparseMatrix + GetContext,
{
    pub unsafe fn from_raw_pointer(raw_pointer: *const T) -> Self {
        Self {
            raw_pointer,
            lifetime_marker: PhantomData,
        }
    }
}

impl<'a, T> Deref for SparseMatrixView<'a, T>
where
    T: GetGraphblasSparseMatrix + GetContext,
{
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.raw_pointer }
    }
}

impl<'a, T> GetGraphblasSparseMatrix for SparseMatrixView<'a, T>
where
    T: GetGraphblasSparseMatrix + GetContext,
{
    unsafe fn graphblas_matrix(&self) -> GrB_Matrix {
        self.deref().graphblas_matrix()
    }

    unsafe fn graphblas_matrix_ref(&self) -> &GrB_Matrix {
        self.deref().graphblas_matrix_ref()
    }

    unsafe fn graphblas_matrix_mut_ref(&mut self) -> &mut GrB_Matrix {
        unimplemented!()
    }
}

impl<'a, T> GetContext for SparseMatrixView<'a, T>
where
    T: GetGraphblasSparseMatrix + GetContext,
{
    fn context(&self) -> Arc<Context> {
        self.deref().context()
    }

    fn context_ref(&self) -> &Arc<Context> {
        self.deref().context_ref()
    }
}
