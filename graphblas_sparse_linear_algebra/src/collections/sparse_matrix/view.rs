use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use suitesparse_graphblas_sys::GrB_Matrix;

use crate::collections::sparse_matrix::{GetGraphblasSparseMatrix, SparseMatrix};
use crate::context::{Context, GetContext};
use crate::value_type::ValueType;

pub struct SparseMatrixView<'a, T: ValueType> {
    raw_pointer: *const SparseMatrix<T>,
    lifetime_marker: PhantomData<&'a SparseMatrix<T>>,
}

impl<'a, T: ValueType> SparseMatrixView<'a, T> {
    pub unsafe fn from_raw_pointer(raw_pointer: *const SparseMatrix<T>) -> Self {
        Self {
            raw_pointer,
            lifetime_marker: PhantomData,
        }
    }
}

impl<'a, T: ValueType> Deref for SparseMatrixView<'a, T> {
    type Target = SparseMatrix<T>;
    fn deref(&self) -> &SparseMatrix<T> {
        unsafe { &*self.raw_pointer }
    }
}

impl<'a, T: ValueType> GetGraphblasSparseMatrix for SparseMatrixView<'a, T> {
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

impl<'a, T: ValueType> GetContext for SparseMatrixView<'a, T> {
    fn context(&self) -> Arc<Context> {
        self.deref().context()
    }

    fn context_ref(&self) -> &Arc<Context> {
        self.deref().context_ref()
    }
}
