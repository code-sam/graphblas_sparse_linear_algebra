use crate::bindings_to_graphblas_implementation::{
    GrB_Matrix_setElement_BOOL, GrB_Matrix_setElement_FP32, GrB_Matrix_setElement_FP64,
    GrB_Matrix_setElement_INT16, GrB_Matrix_setElement_INT32, GrB_Matrix_setElement_INT64,
    GrB_Matrix_setElement_INT8, GrB_Matrix_setElement_UINT16, GrB_Matrix_setElement_UINT32,
    GrB_Matrix_setElement_UINT64, GrB_Matrix_setElement_UINT8,
};
use crate::{
    collections::sparse_matrix::{GraphblasSparseMatrixTrait, MatrixElement, SparseMatrix},
    context::{CallGraphBlasContext, ContextTrait},
    error::SparseLinearAlgebraError,
    index::IndexConversion,
    value_type::{
        utilities_to_implement_traits_for_all_value_types::implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
        ConvertScalar, ValueType,
    },
};

pub trait SetMatrixElement<T: ValueType + SetGraphblasMatrixElement<T>> {
    fn set_element(&mut self, element: MatrixElement<T>) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType + SetGraphblasMatrixElement<T>> SetMatrixElement<T> for SparseMatrix<T> {
    fn set_element(&mut self, element: MatrixElement<T>) -> Result<(), SparseLinearAlgebraError> {
        T::set_graphblas_matrix_element(self, element)
    }
}

pub trait SetGraphblasMatrixElement<T: ValueType> {
    fn set_graphblas_matrix_element(
        matrix: &mut SparseMatrix<T>,
        element: MatrixElement<T>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_set_element_typed {
    ($value_type:ty, $conversion_target_type:ty, $add_element_function:ident) => {
        impl SetGraphblasMatrixElement<$value_type> for $value_type {
            fn set_graphblas_matrix_element(
                matrix: &mut SparseMatrix<$value_type>,
                element: MatrixElement<$value_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let row_index_to_set = element.row_index().to_graphblas_index()?;
                let column_index_to_set = element.column_index().to_graphblas_index()?;
                let context = matrix.context_ref().to_owned();
                let element_value = element.value().to_type()?;
                context.call(
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
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_set_element_typed,
    GrB_Matrix_setElement
);

// impl<T: ValueType + ConvertScalar<T, U> + Copy, U: ValueType> SetMatrixElement<T, U> for SparseMatrix<T> {
//     fn set_element(
//         &mut self,
//         element: MatrixElement<T>,
//     ) -> Result<(), SparseLinearAlgebraError> {
//         let row_index_to_set = element.row_index().to_graphblas_index()?;
//         let column_index_to_set = element.column_index().to_graphblas_index()?;
//         let element_value = element.value().to_type()?;
//         self.context().call(
//             || unsafe {
//                 GrB_Matrix_setElement_BOOL(
//                     self.graphblas_matrix(),
//                     element_value,
//                     row_index_to_set,
//                     column_index_to_set,
//                 )
//             },
//             unsafe {
//                 &self.graphblas_matrix()
//             }
//         )?;
//         Ok(())
//     }
// }

// struct MatrixElementSetter<T: ValueType> {
//     value_type: PhantomData<T>,
// }

// // impl MatrixElementSetter<>

// trait GraphBlasSetMatrixElementFunction<T: ValueType> {
//     fn graphblas_type() -> impl Fn(GrB_Matrix, T, GrB_Index, GrB_Index) -> GrB_Info;
// }
