use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use suitesparse_graphblas_sys::GrB_Vector;

use crate::collections::sparse_vector::{GetGraphblasSparseVector, SparseVector};
use crate::context::{Context, GetContext};
use crate::value_type::ValueType;

pub struct SparseVectorView<'a, T: ValueType> {
    raw_pointer: *const SparseVector<T>,
    lifetime_marker: PhantomData<&'a SparseVector<T>>,
}

impl<'a, T: ValueType> SparseVectorView<'a, T> {
    pub unsafe fn from_raw_pointer(raw_pointer: *const SparseVector<T>) -> Self {
        Self {
            raw_pointer,
            lifetime_marker: PhantomData,
        }
    }
}

impl<'a, T: ValueType> Deref for SparseVectorView<'a, T> {
    type Target = SparseVector<T>;
    fn deref(&self) -> &SparseVector<T> {
        unsafe { &*self.raw_pointer }
    }
}

impl<'a, T: ValueType> GetGraphblasSparseVector for SparseVectorView<'a, T> {
    unsafe fn graphblas_vector(&self) -> GrB_Vector {
        self.deref().graphblas_vector()
    }

    unsafe fn graphblas_vector_ref(&self) -> &GrB_Vector {
        self.deref().graphblas_vector_ref()
    }

    unsafe fn graphblas_vector_mut_ref(&mut self) -> &mut GrB_Vector {
        unimplemented!()
    }
}

impl<'a, T: ValueType> GetContext for SparseVectorView<'a, T> {
    fn context(&self) -> Arc<Context> {
        self.deref().context()
    }

    fn context_ref(&self) -> &Arc<Context> {
        self.deref().context_ref()
    }
}
