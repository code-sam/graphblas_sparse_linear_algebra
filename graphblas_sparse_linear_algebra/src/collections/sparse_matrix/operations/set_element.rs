use crate::collections::sparse_matrix::element::GetMatrixElementCoordinate;
use crate::collections::sparse_matrix::element::GetMatrixElementValue;
use crate::context::CallGraphBlasContext;
use crate::graphblas_bindings::{
    GrB_Matrix_setElement_BOOL, GrB_Matrix_setElement_FP32, GrB_Matrix_setElement_FP64,
    GrB_Matrix_setElement_INT16, GrB_Matrix_setElement_INT32, GrB_Matrix_setElement_INT64,
    GrB_Matrix_setElement_INT8, GrB_Matrix_setElement_UINT16, GrB_Matrix_setElement_UINT32,
    GrB_Matrix_setElement_UINT64, GrB_Matrix_setElement_UINT8,
};
use crate::index::{ElementIndex, IndexConversion};
use crate::value_type::ConvertScalar;
use crate::{
    collections::sparse_matrix::{GetGraphblasSparseMatrix, SparseMatrix},
    context::GetContext,
    error::SparseLinearAlgebraError,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ValueType,
    },
};

pub trait SetSparseMatrixElement<T: ValueType> {
    fn set_matrix_value(
        &mut self,
        row_index: &ElementIndex,
        column_index: &ElementIndex,
        value: T,
    ) -> Result<(), SparseLinearAlgebraError>;
    fn set_matrix_element(
        &mut self,
        element: &(impl GetMatrixElementCoordinate + GetMatrixElementValue<T>),
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType + SetSparseMatrixElementTyped<T>> SetSparseMatrixElement<T> for SparseMatrix<T> {
    fn set_matrix_value(
        &mut self,
        row_index: &ElementIndex,
        column_index: &ElementIndex,
        value: T,
    ) -> Result<(), SparseLinearAlgebraError> {
        T::set_graphblas_matrix_value(self, row_index, column_index, value)
    }

    fn set_matrix_element(
        &mut self,
        element: &(impl GetMatrixElementCoordinate + GetMatrixElementValue<T>),
    ) -> Result<(), SparseLinearAlgebraError> {
        T::set_graphblas_matrix_element(self, element)
    }
}

pub trait SetSparseMatrixElementTyped<T: ValueType> {
    fn set_graphblas_matrix_value(
        matrix: &mut (impl GetGraphblasSparseMatrix + GetContext),
        row_index: &ElementIndex,
        column_index: &ElementIndex,
        value: T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn set_graphblas_matrix_element(
        matrix: &mut (impl GetGraphblasSparseMatrix + GetContext),
        element: &(impl GetMatrixElementCoordinate + GetMatrixElementValue<T>),
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_set_element_typed {
    ($value_type:ty, $conversion_target_type:ty, $add_element_function:ident) => {
        impl SetSparseMatrixElementTyped<$value_type> for $value_type {
            fn set_graphblas_matrix_value(
                matrix: &mut (impl GetGraphblasSparseMatrix + GetContext),
                row_index: &ElementIndex,
                column_index: &ElementIndex,
                value: $value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let row_index_to_set = row_index.as_graphblas_index()?;
                let column_index_to_set = column_index.as_graphblas_index()?;
                let element_value = value.to_type()?;
                matrix.context_ref().call(
                    || unsafe {
                        $add_element_function(
                            matrix.graphblas_matrix(),
                            element_value,
                            row_index_to_set,
                            column_index_to_set,
                        )
                    },
                    unsafe { &matrix.graphblas_matrix() },
                )?;
                Ok(())
            }

            fn set_graphblas_matrix_element(
                matrix: &mut (impl GetGraphblasSparseMatrix + GetContext),
                element: &(impl GetMatrixElementCoordinate + GetMatrixElementValue<$value_type>),
            ) -> Result<(), SparseLinearAlgebraError> {
                <$value_type>::set_graphblas_matrix_value(
                    matrix,
                    element.row_index_ref(),
                    element.column_index_ref(),
                    element.value(),
                )
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_set_element_typed,
    GrB_Matrix_setElement
);
