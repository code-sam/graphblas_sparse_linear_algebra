use std::sync::Arc;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use suitesparse_graphblas_sys::GrB_Matrix_diag;

use crate::collections::sparse_vector::operations::GetSparseVectorLength;
use crate::collections::sparse_vector::GetGraphblasSparseVector;
use crate::collections::sparse_vector::SparseVector;
use crate::context::GetContext;
use crate::error::LogicError;
use crate::graphblas_bindings::{
    GrB_Index, GrB_Matrix_build_BOOL, GrB_Matrix_build_FP32, GrB_Matrix_build_FP64,
    GrB_Matrix_build_INT16, GrB_Matrix_build_INT32, GrB_Matrix_build_INT64, GrB_Matrix_build_INT8,
    GrB_Matrix_build_UINT16, GrB_Matrix_build_UINT32, GrB_Matrix_build_UINT64,
    GrB_Matrix_build_UINT8,
};

use crate::collections::sparse_matrix::sparse_matrix::GetGraphblasSparseMatrix;
use crate::collections::sparse_matrix::SparseMatrix;
use crate::context::CallGraphBlasContext;
use crate::index::DiagonalIndex;
use crate::index::DiagonalIndexConversion;
use crate::index::IndexConversion;
use crate::value_type::ConvertVector;
use crate::{
    collections::sparse_matrix::{MatrixElementList, Size},
    context::Context,
    error::SparseLinearAlgebraError,
    operators::binary_operator::BinaryOperator,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};

pub trait FromDiagonalVector<T: ValueType> {
    fn from_diagonal_vector(
        diagonal: &SparseVector<T>,
        diagonal_index: &DiagonalIndex,
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError>;

    /// The type of the diagonal must match the type of the returned SparseMarix
    fn from_diagonal_vector_untyped(
        diagonal: &(impl GetGraphblasSparseVector + GetContext + GetSparseVectorLength),
        diagonal_index: &DiagonalIndex,
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType> FromDiagonalVector<T> for SparseMatrix<T> {
    /// Returns a square matrix
    fn from_diagonal_vector(
        diagonal: &SparseVector<T>,
        diagonal_index: &DiagonalIndex,
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError> {
        Self::from_diagonal_vector_untyped(diagonal, diagonal_index)
    }

    /// The type of the diagonal must match the type of the returned SparseMarix
    fn from_diagonal_vector_untyped(
        diagonal: &(impl GetGraphblasSparseVector + GetContext + GetSparseVectorLength),
        diagonal_index: &DiagonalIndex,
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError> {
        let diagonal_length = diagonal.length()?;

        let absolute_diagonal_index;
        match TryInto::<usize>::try_into(diagonal_index.abs()) {
            Ok(value) => absolute_diagonal_index = value,
            Err(error) => return Err(LogicError::from(error).into()),
        };

        let row_height = diagonal_length + absolute_diagonal_index;
        let column_width = diagonal_length + absolute_diagonal_index;

        let mut matrix: SparseMatrix<T> =
            SparseMatrix::<T>::new(diagonal.context(), (row_height, column_width).into())?;
        let graphblas_diagonal_index = diagonal_index.as_graphblas_index()?;

        diagonal
            .context_ref()
            .call_without_detailed_error_information(|| unsafe {
                GrB_Matrix_diag(
                    matrix.graphblas_matrix_mut_ref(),
                    diagonal.graphblas_vector(),
                    graphblas_diagonal_index,
                )
            })?;
        return Ok(matrix);
    }
}

pub trait FromMatrixElementList<T: ValueType> {
    fn from_element_list(
        context: Arc<Context>,
        size: Size,
        elements: MatrixElementList<T>,
        reduction_operator_for_duplicates: &impl BinaryOperator<T>,
    ) -> Result<Self, SparseLinearAlgebraError>
    where
        Self: Sized;
}

macro_rules! sparse_matrix_from_element_vector {
    ($value_type:ty, $conversion_target_type: ty, $build_function:ident) => {
        impl FromMatrixElementList<$value_type> for SparseMatrix<$value_type> {
            fn from_element_list(
                context: Arc<Context>,
                size: Size,
                elements: MatrixElementList<$value_type>,
                reduction_operator_for_duplicates: &impl BinaryOperator<$value_type>,
            ) -> Result<Self, SparseLinearAlgebraError> {
                // TODO: check for duplicates
                // TODO: check size constraints
                let matrix = Self::new(context.clone(), size)?;

                let graphblas_row_indices: Vec<GrB_Index> = elements
                    .row_indices_ref()
                    .into_par_iter()
                    .map(|index| index.to_graphblas_index().unwrap())
                    .collect();
                let graphblas_column_indices: Vec<GrB_Index> = elements
                    .column_indices_ref()
                    .into_par_iter()
                    .map(|index| index.to_graphblas_index().unwrap())
                    .collect();

                let element_values = elements.values_ref().to_owned().to_type()?;

                {
                    let number_of_elements = elements.length().to_graphblas_index()?;
                    context.call(
                        || unsafe {
                            $build_function(
                                matrix.graphblas_matrix(),
                                graphblas_row_indices.as_ptr(),
                                graphblas_column_indices.as_ptr(),
                                element_values.as_ptr(),
                                number_of_elements,
                                reduction_operator_for_duplicates.graphblas_type(),
                            )
                        },
                        unsafe { matrix.graphblas_matrix_ref() },
                    )?;
                }
                Ok(matrix)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    sparse_matrix_from_element_vector,
    GrB_Matrix_build
);
