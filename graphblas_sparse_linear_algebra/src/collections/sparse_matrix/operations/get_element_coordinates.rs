use suitesparse_graphblas_sys::{
    GrB_Matrix_extractTuples_BOOL, GrB_Matrix_extractTuples_FP32, GrB_Matrix_extractTuples_FP64,
    GrB_Matrix_extractTuples_INT16, GrB_Matrix_extractTuples_INT32, GrB_Matrix_extractTuples_INT64,
    GrB_Matrix_extractTuples_INT8, GrB_Matrix_extractTuples_UINT16,
    GrB_Matrix_extractTuples_UINT32, GrB_Matrix_extractTuples_UINT64,
    GrB_Matrix_extractTuples_UINT8,
};

use crate::collections::collection::Collection;
use crate::collections::sparse_matrix::ColumnIndex;
use crate::collections::sparse_matrix::CoordinateList;
use crate::collections::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_matrix::RowIndex;
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

pub trait GetSparseMatrixCoordinates {
    fn element_coordinates(&self) -> Result<CoordinateList, SparseLinearAlgebraError>;
    fn element_row_indices(&self) -> Result<Vec<RowIndex>, SparseLinearAlgebraError>;
    fn element_column_indices(&self) -> Result<Vec<ColumnIndex>, SparseLinearAlgebraError>;
}

impl<T: ValueType + GetSparseMatrixCoordinatesTyped> GetSparseMatrixCoordinates
    for SparseMatrix<T>
{
    fn element_coordinates(&self) -> Result<CoordinateList, SparseLinearAlgebraError> {
        T::get_coordinates(self)
    }

    fn element_row_indices(&self) -> Result<Vec<RowIndex>, SparseLinearAlgebraError> {
        T::get_row_indices(self)
    }

    fn element_column_indices(&self) -> Result<Vec<ColumnIndex>, SparseLinearAlgebraError> {
        T::get_column_indices(self)
    }
}

pub trait GetSparseMatrixCoordinatesTyped {
    fn get_coordinates(
        matrix: &(impl GetGraphblasSparseMatrix + Collection),
    ) -> Result<CoordinateList, SparseLinearAlgebraError>;
    fn get_row_indices(
        matrix: &(impl GetGraphblasSparseMatrix + Collection),
    ) -> Result<Vec<RowIndex>, SparseLinearAlgebraError>;
    fn get_column_indices(
        matrix: &(impl GetGraphblasSparseMatrix + Collection),
    ) -> Result<Vec<ColumnIndex>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_coordinates_typed {
    ($value_type:ty, $_graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetSparseMatrixCoordinatesTyped for $value_type {
            fn get_coordinates(
                matrix: &(impl GetGraphblasSparseMatrix + Collection),
            ) -> Result<CoordinateList, SparseLinearAlgebraError> {
                let number_of_stored_elements = matrix.number_of_stored_elements()?;

                let mut row_indices = Vec::with_capacity(number_of_stored_elements);
                let mut column_indices = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                matrix.context_ref().call(|| unsafe {
                    $get_element_function(
                        row_indices.as_mut_ptr(),
                        column_indices.as_mut_ptr(),
                        std::ptr::null_mut(),
                        &mut number_of_stored_and_returned_elements,
                        matrix.graphblas_matrix_ptr(),
                    )
                }, unsafe{ &matrix.graphblas_matrix_ptr() } )?;

                let number_of_returned_elements = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if number_of_returned_elements == number_of_stored_elements {
                        row_indices.set_len(number_of_returned_elements);
                        column_indices.set_len(number_of_returned_elements);
                    } else {
                        let err: SparseLinearAlgebraError = GraphblasError::new(GraphblasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values {}",number_of_stored_elements, number_of_returned_elements)).into();
                        return Err(err)
                    }
                };

                let row_indices = row_indices.to_type()?;
                let column_indices = column_indices.to_type()?;

                let coordinate_list = CoordinateList::from_vectors(row_indices, column_indices)?;

                Ok(coordinate_list)
            }

            fn get_row_indices(
                matrix: &(impl GetGraphblasSparseMatrix + Collection),
            ) -> Result<Vec<RowIndex>, SparseLinearAlgebraError> {
                let number_of_stored_elements = matrix.number_of_stored_elements()?;

                let mut row_indices = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                matrix.context_ref().call(|| unsafe {
                    $get_element_function(
                        row_indices.as_mut_ptr(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        &mut number_of_stored_and_returned_elements,
                        matrix.graphblas_matrix_ptr(),
                    )
                }, unsafe{ &matrix.graphblas_matrix_ptr() } )?;

                let number_of_returned_elements = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if number_of_returned_elements == number_of_stored_elements {
                        row_indices.set_len(number_of_returned_elements);
                    } else {
                        let err: SparseLinearAlgebraError = GraphblasError::new(GraphblasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values {}",number_of_stored_elements, number_of_returned_elements)).into();
                        return Err(err)
                    }
                };

                let row_indices = row_indices.to_type()?;

                Ok(row_indices)
            }

            fn get_column_indices(
                matrix: &(impl GetGraphblasSparseMatrix + Collection),
            ) -> Result<Vec<RowIndex>, SparseLinearAlgebraError> {
                let number_of_stored_elements = matrix.number_of_stored_elements()?;

                let mut column_indices = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                matrix.context_ref().call(|| unsafe {
                    $get_element_function(
                        std::ptr::null_mut(),
                        column_indices.as_mut_ptr(),
                        std::ptr::null_mut(),
                        &mut number_of_stored_and_returned_elements,
                        matrix.graphblas_matrix_ptr(),
                    )
                }, unsafe{ &matrix.graphblas_matrix_ptr() } )?;

                let number_of_returned_elements = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if number_of_returned_elements == number_of_stored_elements {
                        column_indices.set_len(number_of_returned_elements);
                    } else {
                        let err: SparseLinearAlgebraError = GraphblasError::new(GraphblasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values {}",number_of_stored_elements, number_of_returned_elements)).into();
                        return Err(err)
                    }
                };

                let column_indices = column_indices.to_type()?;

                Ok(column_indices)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_coordinates_typed,
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
            matrix.element_row_indices().unwrap(),
            element_list.row_indices_ref().to_vec()
        );
        assert_eq!(
            matrix.element_column_indices().unwrap(),
            element_list.column_indices_ref().to_vec()
        );
    }
}
