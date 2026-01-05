use std::{marker::PhantomData, sync::Arc};

use suitesparse_graphblas_sys::GrB_Vector;

use crate::collections::sparse_vector::{GetGraphblasSparseVector, SparseVector};
use crate::context::{Context, GetContext};
use crate::value_type::ValueType;

pub struct GraphblasVectorHandleUntyped<'a> {
    pointer_to_graphblas_vector: GrB_Vector,
    context: Arc<Context>,
    _lifetime_marker: PhantomData<&'a GrB_Vector>,
}

pub struct GraphblasMatrixHandle<'a, T: ValueType> {
    pointer_to_graphblas_vector: GrB_Vector,
    context: Arc<Context>,
    _lifetime_and_value_type_marker: PhantomData<&'a T>,
}

impl<'a> GraphblasVectorHandleUntyped<'a> {
    pub fn from_sparse_vector(
        sparse_vector: &'a (impl GetGraphblasSparseVector + GetContext),
    ) -> Self {
        Self {
            pointer_to_graphblas_vector: unsafe { sparse_vector.graphblas_vector_ptr() },
            context: sparse_vector.context(),
            _lifetime_marker: PhantomData,
        }
    }
}

impl<'a, T: ValueType> GraphblasMatrixHandle<'a, T> {
    pub fn from_sparse_matrix(sparse_vector: &SparseVector<T>) -> Self {
        Self {
            pointer_to_graphblas_vector: unsafe { sparse_vector.graphblas_vector_ptr() },
            context: sparse_vector.context(),
            _lifetime_and_value_type_marker: PhantomData,
        }
    }
}

impl<'a> GetGraphblasSparseVector for GraphblasVectorHandleUntyped<'a> {
    unsafe fn graphblas_vector_ptr(&self) -> GrB_Vector {
        self.pointer_to_graphblas_vector
    }

    unsafe fn graphblas_vector_ptr_ref(&self) -> &GrB_Vector {
        &self.pointer_to_graphblas_vector
    }

    unsafe fn graphblas_vector_ptr_mut(&mut self) -> &mut GrB_Vector {
        &mut self.pointer_to_graphblas_vector
    }
}

impl<'a> GetContext for GraphblasVectorHandleUntyped<'a> {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }

    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl<'a, T: ValueType> GetGraphblasSparseVector for GraphblasMatrixHandle<'a, T> {
    unsafe fn graphblas_vector_ptr(&self) -> GrB_Vector {
        self.pointer_to_graphblas_vector
    }

    unsafe fn graphblas_vector_ptr_ref(&self) -> &GrB_Vector {
        &self.pointer_to_graphblas_vector
    }

    unsafe fn graphblas_vector_ptr_mut(&mut self) -> &mut GrB_Vector {
        &mut self.pointer_to_graphblas_vector
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
