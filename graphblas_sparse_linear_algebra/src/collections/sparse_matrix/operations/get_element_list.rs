use suitesparse_graphblas_sys::GrB_Index;
use suitesparse_graphblas_sys::{
    GrB_Matrix_extractTuples_BOOL, GrB_Matrix_extractTuples_FP32, GrB_Matrix_extractTuples_FP64,
    GrB_Matrix_extractTuples_INT16, GrB_Matrix_extractTuples_INT32, GrB_Matrix_extractTuples_INT64,
    GrB_Matrix_extractTuples_INT8, GrB_Matrix_extractTuples_UINT16,
    GrB_Matrix_extractTuples_UINT32, GrB_Matrix_extractTuples_UINT64,
    GrB_Matrix_extractTuples_UINT8,
};

use crate::collections::collection::Collection;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_matrix::MatrixElementList;
use crate::collections::sparse_matrix::SparseMatrix;
use crate::context::CallGraphBlasContext;
use crate::error::GraphblasError;
use crate::error::GraphblasErrorType;
use crate::error::SparseLinearAlgebraError;
use crate::index::ElementIndex;
use crate::index::IndexConversion;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::ConvertVector;
use crate::value_type::ValueType;

pub trait GetSparseMatrixElementList<T: ValueType> {
    fn element_list(&self) -> Result<MatrixElementList<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType + GetSparseMatrixElementListTyped<T>> GetSparseMatrixElementList<T>
    for SparseMatrix<T>
{
    fn element_list(&self) -> Result<MatrixElementList<T>, SparseLinearAlgebraError> {
        T::get_element_list(self)
    }
}

pub trait GetSparseMatrixElementListTyped<T: ValueType> {
    fn get_element_list(
        matrix: &(impl GetGraphblasSparseMatrix + Collection),
    ) -> Result<MatrixElementList<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_list {
    ($value_type:ty, $_graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetSparseMatrixElementListTyped<$value_type> for $value_type {
            fn get_element_list(
                matrix: &(impl GetGraphblasSparseMatrix + Collection),
            ) -> Result<MatrixElementList<$value_type>, SparseLinearAlgebraError> {
                let number_of_stored_elements = matrix.number_of_stored_elements()?;

                let mut row_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);
                let mut column_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);
                let mut values = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                matrix.context_ref().call(|| unsafe {
                    $get_element_function(
                        row_indices.as_mut_ptr(),
                        column_indices.as_mut_ptr(),
                        values.as_mut_ptr(),
                        &mut number_of_stored_and_returned_elements,
                        matrix.graphblas_matrix_ptr(),
                    )
                }, unsafe{ &matrix.graphblas_matrix_ptr() } )?;

                let number_of_returned_elements = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if number_of_returned_elements == number_of_stored_elements {
                        row_indices.set_len(number_of_returned_elements);
                        column_indices.set_len(number_of_returned_elements);
                        values.set_len(number_of_returned_elements);
                    } else {
                        let err: SparseLinearAlgebraError = GraphblasError::new(GraphblasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values{}",number_of_stored_elements, number_of_returned_elements)).into();
                        return Err(err)
                    }
                };

                let row_indices = row_indices.to_type()?;
                let column_indices = column_indices.to_type()?;

                let values = values.to_type()?;

                let element_list = MatrixElementList::from_vectors(row_indices, column_indices, values)?;
                Ok(element_list)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_list,
    GrB_Matrix_extractTuples
);
