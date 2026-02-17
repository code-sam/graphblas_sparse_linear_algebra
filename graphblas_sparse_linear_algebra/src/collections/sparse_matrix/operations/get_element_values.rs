use suitesparse_graphblas_sys::{
    GrB_Matrix_extractTuples_BOOL, GrB_Matrix_extractTuples_FP32, GrB_Matrix_extractTuples_FP64,
    GrB_Matrix_extractTuples_INT16, GrB_Matrix_extractTuples_INT32, GrB_Matrix_extractTuples_INT64,
    GrB_Matrix_extractTuples_INT8, GrB_Matrix_extractTuples_UINT16,
    GrB_Matrix_extractTuples_UINT32, GrB_Matrix_extractTuples_UINT64,
    GrB_Matrix_extractTuples_UINT8,
};

use crate::collections::collection::Collection;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_matrix::SparseMatrix;
use crate::context::CallGraphBlasContext;
use crate::context::GetContext;
use crate::error::GraphblasError;
use crate::error::GraphblasErrorType;
use crate::error::SparseLinearAlgebraError;
use crate::index::ElementIndex;
use crate::index::IndexConversion;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type;
use crate::value_type::ConvertVector;
use crate::value_type::ValueType;

pub trait GetSparseMatrixElementValues<T: ValueType> {
    fn element_values(&self) -> Result<Vec<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType + GetSparseMatrixElementValuesTyped<T>> GetSparseMatrixElementValues<T>
    for SparseMatrix<T>
{
    fn element_values(&self) -> Result<Vec<T>, SparseLinearAlgebraError> {
        T::get_element_values(self)
    }
}

pub trait GetSparseMatrixElementValuesTyped<T: ValueType> {
    fn get_element_values(matrix: &SparseMatrix<T>) -> Result<Vec<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_values {
    ($value_type:ty, $_graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetSparseMatrixElementValuesTyped<$value_type> for $value_type {
            fn get_element_values(
                matrix: &SparseMatrix<$value_type>,
            ) -> Result<Vec<$value_type>, SparseLinearAlgebraError> {
                let number_of_stored_elements = matrix.number_of_stored_elements()?;

                let mut values = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                matrix.context_ref().call(|| unsafe {
                    $get_element_function(
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        values.as_mut_ptr(),
                        &mut number_of_stored_and_returned_elements,
                        matrix.graphblas_matrix_ptr(),
                    )
                }, unsafe{ &matrix.graphblas_matrix_ptr() } )?;

                let number_of_returned_elements = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if number_of_returned_elements == number_of_stored_elements {
                        values.set_len(number_of_returned_elements);
                    } else {
                        let err: SparseLinearAlgebraError = GraphblasError::new(GraphblasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values{}",number_of_stored_elements, number_of_returned_elements)).into();
                        return Err(err)
                    }
                };

                let values = values.to_type()?;

                Ok(values)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_values,
    GrB_Matrix_extractTuples
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::operations::FromMatrixElementList;
    use crate::collections::sparse_matrix::{MatrixElementList, SparseMatrix};
    use crate::collections::Collection;
    use crate::context::Context;
    use crate::operators::binary_operator::First;

    #[test]
    fn get_element_values_from_matrix() {
        let context = Context::init_default().unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (2, 5, 11).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            context.clone(),
            (10, 15).into(),
            element_list.clone(),
            &First::<u8>::new(),
        )
        .unwrap();

        // println!("original element list: {:?}", element_list);
        // println!(
        //     "stored element list: {:?}",
        //     matrix.get_element_list().unwrap()
        // );
        assert_eq!(
            matrix.number_of_stored_elements().unwrap(),
            element_list.length()
        );

        assert_eq!(
            matrix.element_values().unwrap(),
            element_list.values_ref().to_vec()
        );
    }
}
