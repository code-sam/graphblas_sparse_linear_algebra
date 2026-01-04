use std::{marker::PhantomData, sync::Arc};

use suitesparse_graphblas_sys::GrB_Matrix;

use crate::collections::sparse_matrix::{GetGraphblasSparseMatrix, SparseMatrix};
use crate::context::{Context, GetContext};
use crate::value_type::ValueType;

pub struct GraphblasMatrixHandleUntyped<'a> {
    pointer_to_graphblas_matrix: GrB_Matrix,
    context: Arc<Context>,
    _lifetime_marker: PhantomData<&'a GrB_Matrix>,
}

pub struct GraphblasMatrixHandle<'a, T: ValueType> {
    pointer_to_graphblas_matrix: GrB_Matrix,
    context: Arc<Context>,
    _lifetime_and_value_type_marker: PhantomData<&'a T>,
}

impl<'a> GraphblasMatrixHandleUntyped<'a> {
    pub fn from_sparse_matrix(
        sparse_matrix: &'a (impl GetGraphblasSparseMatrix + GetContext),
    ) -> Self {
        Self {
            pointer_to_graphblas_matrix: unsafe { sparse_matrix.graphblas_matrix_ptr() },
            context: sparse_matrix.context(),
            _lifetime_marker: PhantomData,
        }
    }
}

impl<'a, T: ValueType> GraphblasMatrixHandle<'a, T> {
    pub fn from_sparse_matrix(sparse_matrix: &SparseMatrix<T>) -> Self {
        Self {
            pointer_to_graphblas_matrix: unsafe { sparse_matrix.graphblas_matrix_ptr() },
            context: sparse_matrix.context(),
            _lifetime_and_value_type_marker: PhantomData,
        }
    }
}

impl<'a> GetGraphblasSparseMatrix for GraphblasMatrixHandleUntyped<'a> {
    unsafe fn graphblas_matrix_ptr(&self) -> GrB_Matrix {
        self.pointer_to_graphblas_matrix
    }

    unsafe fn graphblas_matrix_ptr_ref(&self) -> &GrB_Matrix {
        &self.pointer_to_graphblas_matrix
    }

    unsafe fn graphblas_matrix_ptr_mut(&mut self) -> &mut GrB_Matrix {
        &mut self.pointer_to_graphblas_matrix
    }
}

impl<'a> GetContext for GraphblasMatrixHandleUntyped<'a> {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl<'a, T: ValueType> GetGraphblasSparseMatrix for GraphblasMatrixHandle<'a, T> {
    unsafe fn graphblas_matrix_ptr(&self) -> GrB_Matrix {
        self.pointer_to_graphblas_matrix
    }

    unsafe fn graphblas_matrix_ptr_ref(&self) -> &GrB_Matrix {
        &self.pointer_to_graphblas_matrix
    }

    unsafe fn graphblas_matrix_ptr_mut(&mut self) -> &mut GrB_Matrix {
        &mut self.pointer_to_graphblas_matrix
    }
}

impl<'a, T: ValueType> GetContext for GraphblasMatrixHandle<'a, T> {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}
