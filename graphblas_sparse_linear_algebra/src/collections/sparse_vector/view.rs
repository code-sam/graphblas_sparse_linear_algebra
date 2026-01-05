use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use suitesparse_graphblas_sys::GrB_Vector;

use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::context::{Context, GetContext};

pub struct SparseVectorView<'a, T>
where
    T: GetGraphblasSparseVector + GetContext,
{
    raw_pointer: *const T,
    lifetime_marker: PhantomData<&'a T>,
}

impl<'a, T> SparseVectorView<'a, T>
where
    T: GetGraphblasSparseVector + GetContext,
{
    pub unsafe fn from_raw_pointer(raw_pointer: *const T) -> Self {
        Self {
            raw_pointer,
            lifetime_marker: PhantomData,
        }
    }
}

impl<'a, T> Deref for SparseVectorView<'a, T>
where
    T: GetGraphblasSparseVector + GetContext,
{
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.raw_pointer }
    }
}

impl<'a, T> GetGraphblasSparseVector for SparseVectorView<'a, T>
where
    T: GetGraphblasSparseVector + GetContext,
{
    unsafe fn graphblas_vector_ptr(&self) -> GrB_Vector {
        self.deref().graphblas_vector_ptr()
    }

    unsafe fn graphblas_vector_ptr_ref(&self) -> &GrB_Vector {
        self.deref().graphblas_vector_ptr_ref()
    }

    unsafe fn graphblas_vector_ptr_mut(&mut self) -> &mut GrB_Vector {
        unimplemented!()
    }
}

impl<'a, T> GetContext for SparseVectorView<'a, T>
where
    T: GetGraphblasSparseVector + GetContext,
{
    fn context(&self) -> Arc<Context> {
        self.deref().context()
    }

    fn context_ref(&self) -> &Arc<Context> {
        self.deref().context_ref()
    }
}
