use std::ffi::c_void;
use std::mem::size_of;

use suitesparse_graphblas_sys::GxB_Vector_pack_Full;

use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::operators::options::OperatorOptions;
use crate::value_types::value_type::BuiltInValueType;
use crate::value_types::value_type::{ConvertScalar, ConvertVector};
use crate::{
    collections::sparse_vector::SparseVector, error::SparseLinearAlgebraError,
    value_types::value_type::ValueType,
};

use super::FullVectorData;
use crate::collections::sparse_vector::data::full::FullVectorDataTrait;

pub trait VectorData<T: ValueType> {
    fn import_full_vector_data(
        &mut self,
        vector_data: &mut FullVectorData<T>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType + BuiltInValueType> VectorData<T> for SparseVector<T> {
    fn import_full_vector_data(
        &mut self,
        vector_data: &mut FullVectorData<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let length = size_of::<[T; vector_data.values_ref().len()]>.to_type()?;
        // let values = vector_data.values().to_type()?;
        self.context_ref().call(
            || unsafe {
                GxB_Vector_pack_Full(
                    self.graphblas_vector(),
                    vector_data.values_mut_ref().as_mut_ptr() as *mut *mut c_void,
                    // vector_data.values_mut_slice().as_mut_ptr() as *mut *mut c_void,
                    // values.as_mut_ptr(),
                    length,
                    false,
                    OperatorOptions::new_default().graphblas_descriptor(),
                )
            },
            self.graphblas_vector_ref(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        collections::sparse_vector::{data::FullVectorData, SparseVector},
        context::{Context, Mode},
    };

    use super::VectorData;

    #[test]
    fn import_full_vector_data() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let mut vector_data = FullVectorData::<i16>::from_vector(vec![0, 1, 2, 3]);

        let mut sparse_vector = SparseVector::<i16>::new(&context, &4).unwrap();

        sparse_vector
            .import_full_vector_data(&mut vector_data)
            .unwrap()
    }
}
