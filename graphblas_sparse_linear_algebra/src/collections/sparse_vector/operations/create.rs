use std::sync::Arc;

use crate::graphblas_bindings::{
    GrB_Vector_build_BOOL, GrB_Vector_build_FP32, GrB_Vector_build_FP64, GrB_Vector_build_INT16,
    GrB_Vector_build_INT32, GrB_Vector_build_INT64, GrB_Vector_build_INT8, GrB_Vector_build_UINT16,
    GrB_Vector_build_UINT32, GrB_Vector_build_UINT64, GrB_Vector_build_UINT8,
};

use crate::collections::sparse_vector::sparse_vector::GetGraphblasSparseVector;
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::index::IndexConversion;
use crate::value_type::ConvertVector;
use crate::{
    collections::sparse_vector::{SparseVector, VectorElementList},
    context::Context,
    error::SparseLinearAlgebraError,
    index::ElementIndex,
    operators::binary_operator::BinaryOperator,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};

pub trait FromVectorElementList<T: ValueType> {
    fn from_element_list(
        context: Arc<Context>,
        lenth: ElementIndex,
        elements: VectorElementList<T>,
        reduction_operator_for_duplicates: &impl BinaryOperator<T>,
    ) -> Result<SparseVector<T>, SparseLinearAlgebraError>;
}

macro_rules! sparse_matrix_from_element_vector {
    ($value_type:ty, $graphblas_implementation_type:ty, $build_function:ident) => {
        impl FromVectorElementList<$value_type> for SparseVector<$value_type> {
            fn from_element_list(
                context: Arc<Context>,
                length: ElementIndex,
                elements: VectorElementList<$value_type>,
                reduction_operator_for_duplicates: &impl BinaryOperator<$value_type>,
            ) -> Result<Self, SparseLinearAlgebraError> {
                // TODO: check for duplicates
                // TODO: check size constraints
                let vector = Self::new(context, length)?;

                let mut graphblas_indices = Vec::with_capacity(elements.length());

                for i in 0..elements.length() {
                    graphblas_indices.push(elements.index(i)?.to_graphblas_index()?);
                }
                let number_of_elements = elements.length().to_graphblas_index()?;
                let element_values = elements.values_ref().to_owned().to_type()?;
                vector.context_ref().call(
                    || unsafe {
                        $build_function(
                            vector.graphblas_vector_ptr(),
                            graphblas_indices.as_ptr(),
                            element_values.as_ptr(),
                            number_of_elements,
                            reduction_operator_for_duplicates.graphblas_type(),
                        )
                    },
                    unsafe { &vector.graphblas_vector_ptr() },
                )?;
                Ok(vector)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    sparse_matrix_from_element_vector,
    GrB_Vector_build
);
