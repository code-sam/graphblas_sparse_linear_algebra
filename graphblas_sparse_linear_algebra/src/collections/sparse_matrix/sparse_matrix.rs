use std::convert::TryInto;
use std::marker::{PhantomData, Send, Sync};
use std::mem::MaybeUninit;
use std::sync::Arc;

use rayon::prelude::*;

use suitesparse_graphblas_sys::{GrB_Matrix_diag, GxB_Matrix_isStoredElement};

use crate::bindings_to_graphblas_implementation::{
    GrB_Index, GrB_Matrix, GrB_Matrix_build_BOOL, GrB_Matrix_build_FP32, GrB_Matrix_build_FP64,
    GrB_Matrix_build_INT16, GrB_Matrix_build_INT32, GrB_Matrix_build_INT64, GrB_Matrix_build_INT8,
    GrB_Matrix_build_UINT16, GrB_Matrix_build_UINT32, GrB_Matrix_build_UINT64,
    GrB_Matrix_build_UINT8, GrB_Matrix_clear, GrB_Matrix_dup, GrB_Matrix_extractTuples_BOOL,
    GrB_Matrix_extractTuples_FP32, GrB_Matrix_extractTuples_FP64, GrB_Matrix_extractTuples_INT16,
    GrB_Matrix_extractTuples_INT32, GrB_Matrix_extractTuples_INT64, GrB_Matrix_extractTuples_INT8,
    GrB_Matrix_extractTuples_UINT16, GrB_Matrix_extractTuples_UINT32,
    GrB_Matrix_extractTuples_UINT64, GrB_Matrix_extractTuples_UINT8, GrB_Matrix_free,
    GrB_Matrix_ncols, GrB_Matrix_new, GrB_Matrix_nrows, GrB_Matrix_nvals, GrB_Matrix_removeElement,
    GrB_Matrix_resize,
};
use crate::collections::collection::Collection;
use crate::collections::sparse_vector::{
    GraphblasSparseVectorTrait, SparseVector, SparseVectorTrait,
};
use crate::error::{
    GraphBlasError, GraphBlasErrorType, LogicError, LogicErrorType, SparseLinearAlgebraError,
    SparseLinearAlgebraErrorType,
};
use crate::operators::mask::MatrixMask;
use crate::collections::sparse_matrix::operations::GetMatrixElementValue;

use super::coordinate::Coordinate;
use super::element::{MatrixElement, MatrixElementList};
use super::size::Size;

use crate::context::ContextTrait;
use crate::context::{CallGraphBlasContext, Context};
use crate::operators::binary_operator::BinaryOperator;

use crate::index::{DiagonalIndex, DiagonalIndexConversion, ElementIndex, IndexConversion};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
    implement_macro_for_all_value_types,
    implement_macro_for_all_value_types_and_graphblas_function,
};
use crate::value_type::{ConvertScalar, ConvertVector, ValueType};

// static DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS: Lazy<OperatorOptions> =
//     Lazy::new(|| OperatorOptions::new_default());

pub type ColumnIndex = ElementIndex;
pub type RowIndex = ElementIndex;

#[derive(Debug)]
pub struct SparseMatrix<T: ValueType> {
    context: Arc<Context>,
    matrix: GrB_Matrix,
    value_type: PhantomData<T>,
}

// Mutable access to GrB_Matrix shall occur through a write lock on RwLock<GrB_Matrix>.
// Code review must consider that the correct lock is made via
// SparseMatrix::get_write_lock() and SparseMatrix::get_read_lock().
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<T: ValueType> Send for SparseMatrix<T> {}
unsafe impl<T: ValueType> Sync for SparseMatrix<T> {}

impl<T: ValueType> SparseMatrix<T> {
    pub fn new(context: &Arc<Context>, size: &Size) -> Result<Self, SparseLinearAlgebraError> {
        let mut matrix: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
        let context = context.to_owned();

        let row_height = size.row_height().to_graphblas_index()?;
        let column_width = size.column_width().to_graphblas_index()?;

        context.call_without_detailed_error_information(|| unsafe {
            GrB_Matrix_new(
                matrix.as_mut_ptr(),
                <T>::to_graphblas_type(),
                row_height,
                column_width,
            )
        })?;

        let matrix = unsafe { matrix.assume_init() };
        return Ok(SparseMatrix {
            context,
            matrix: matrix,
            value_type: PhantomData,
        });
    }

    // TODO
    // fn from_matrices(matrices: Vec<SparseMatrix<T>, >) -> Result<Self, SparseLinearAlgebraError> {

    // }
}

impl<T: ValueType> ContextTrait for SparseMatrix<T> {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }
    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl<T: ValueType> Collection for SparseMatrix<T> {
    /// remove all elements in th matrix
    fn clear(&mut self) -> Result<(), SparseLinearAlgebraError> {
        self.context
            .call(|| unsafe { GrB_Matrix_clear(self.matrix) }, &self.matrix)?;
        Ok(())
    }

    fn number_of_stored_elements(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut number_of_values: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context.call(
            || unsafe { GrB_Matrix_nvals(number_of_values.as_mut_ptr(), self.matrix) },
            &self.matrix,
        )?;
        let number_of_values = unsafe { number_of_values.assume_init() };
        Ok(ElementIndex::from_graphblas_index(number_of_values)?)
    }
}

pub trait GraphblasSparseMatrixTrait {
    unsafe fn graphblas_matrix(&self) -> GrB_Matrix;
    unsafe fn graphblas_matrix_ref(&self) -> &GrB_Matrix;
    unsafe fn graphblas_matrix_mut_ref(&mut self) -> &mut GrB_Matrix;
}

impl<T: ValueType> GraphblasSparseMatrixTrait for SparseMatrix<T> {
    unsafe fn graphblas_matrix(&self) -> GrB_Matrix {
        self.matrix
    }

    unsafe fn graphblas_matrix_ref(&self) -> &GrB_Matrix {
        &self.matrix
    }

    unsafe fn graphblas_matrix_mut_ref(&mut self) -> &mut GrB_Matrix {
        &mut self.matrix
    }
}

pub trait SparseMatrixTrait {
    fn column_width(&self) -> Result<ElementIndex, SparseLinearAlgebraError>;
    fn drop_element(&mut self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError>;
    fn is_element(&self, coordinate: Coordinate) -> Result<bool, SparseLinearAlgebraError>;
    fn try_is_element(&self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError>;

    /// All elements of self with an index coordinate outside of the new size are dropped.
    fn resize(&mut self, new_size: &Size) -> Result<(), SparseLinearAlgebraError>;
    fn row_height(&self) -> Result<ElementIndex, SparseLinearAlgebraError>;
    fn size(&self) -> Result<Size, SparseLinearAlgebraError>;
}

impl<T: ValueType> SparseMatrixTrait for SparseMatrix<T> {
    fn column_width(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut column_width: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context.call(
            || unsafe { GrB_Matrix_ncols(column_width.as_mut_ptr(), self.matrix) },
            &self.matrix,
        )?;
        let column_width = unsafe { column_width.assume_init() };
        Ok(ElementIndex::from_graphblas_index(column_width)?)
    }

    fn drop_element(&mut self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError> {
        let row_index_to_delete = coordinate.row_index().to_graphblas_index()?;
        let column_index_to_delete = coordinate.column_index().to_graphblas_index()?;

        let context = self.context.to_owned();
        context.call(
            || unsafe {
                GrB_Matrix_removeElement(self.matrix, row_index_to_delete, column_index_to_delete)
            },
            &self.matrix,
        )?;
        Ok(())
    }

    fn is_element(&self, coordinate: Coordinate) -> Result<bool, SparseLinearAlgebraError> {
        let row_index = coordinate.row_index().to_graphblas_index()?;
        let column_index = coordinate.column_index().to_graphblas_index()?;

        let context = self.context.to_owned();
        let result = context.call(
            || unsafe { GxB_Matrix_isStoredElement(self.matrix, row_index, column_index) },
            &self.matrix,
        );
        match result {
            Ok(_) => Ok(true),
            Err(error) => match error.error_type() {
                SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                    GraphBlasErrorType::NoValue,
                )) => Ok(false),
                _ => Err(error),
            },
        }
    }

    fn try_is_element(&self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError> {
        let row_index = coordinate.row_index().to_graphblas_index()?;
        let column_index = coordinate.column_index().to_graphblas_index()?;

        let context = self.context.to_owned();
        let result = context.call(
            || unsafe { GxB_Matrix_isStoredElement(self.matrix, row_index, column_index) },
            &self.matrix,
        );
        match result {
            Ok(_) => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// All elements of self with an index coordinate outside of the new size are dropped.
    fn resize(&mut self, new_size: &Size) -> Result<(), SparseLinearAlgebraError> {
        let new_row_height = new_size.row_height().to_graphblas_index()?;
        let new_column_width = new_size.column_width().to_graphblas_index()?;

        let context = self.context.to_owned();
        context.call(
            || unsafe { GrB_Matrix_resize(self.matrix, new_row_height, new_column_width) },
            &self.matrix,
        )?;
        Ok(())
    }

    fn row_height(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut row_height: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context.call(
            || unsafe { GrB_Matrix_nrows(row_height.as_mut_ptr(), self.matrix) },
            &self.matrix,
        )?;
        let row_height = unsafe { row_height.assume_init() };
        Ok(ElementIndex::from_graphblas_index(row_height)?)
    }

    fn size(&self) -> Result<Size, SparseLinearAlgebraError> {
        Ok(Size::new(self.row_height()?, self.column_width()?))
    }
}

impl<T: ValueType> Drop for SparseMatrix<T> {
    fn drop(&mut self) -> () {
        let context = self.context.to_owned();
        let _ = context.call_without_detailed_error_information(|| unsafe {
            GrB_Matrix_free(&mut self.matrix)
        });
    }
}

impl<T: ValueType> Clone for SparseMatrix<T> {
    fn clone(&self) -> Self {
        let mut matrix_copy: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
        self.context
            .call(
                || unsafe { GrB_Matrix_dup(matrix_copy.as_mut_ptr(), self.matrix) },
                &self.matrix,
            )
            .unwrap();

        SparseMatrix {
            context: self.context.to_owned(),
            matrix: unsafe { matrix_copy.assume_init() },
            value_type: PhantomData,
        }
    }
}

// TODO improve printing format
// summary data, column aligning
macro_rules! implement_display {
    ($value_type:ty) => {
        impl std::fmt::Display for SparseMatrix<$value_type> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let element_list: MatrixElementList<$value_type>;
                match self.get_element_list() {
                    Err(_error) => return Err(std::fmt::Error),
                    Ok(list) => {
                        element_list = list;
                    }
                }

                let row_indices = element_list.row_indices_ref();
                let column_indices = element_list.column_indices_ref();
                let values = element_list.values_ref();

                writeln! {f,"Matrix size: {:?}", self.size()?};
                writeln! {f,"Number of stored elements: {:?}", self.number_of_stored_elements()?};

                for element_index in 0..values.len() {
                    write!(
                        f,
                        "({}, {}, {})\n",
                        row_indices[element_index],
                        column_indices[element_index],
                        values[element_index]
                    );
                }
                return writeln!(f, "");
            }
        }
    };
}

implement_macro_for_all_value_types!(implement_display);

pub trait FromDiagonalVector<T: ValueType> {
    fn from_diagonal_vector(
        context: &Arc<Context>,
        diagonal: &SparseVector<T>,
        diagonal_index: &DiagonalIndex,
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError>;
}

impl<T: ValueType> FromDiagonalVector<T> for SparseMatrix<T> {
    /// Returns a square matrix
    fn from_diagonal_vector(
        context: &Arc<Context>,
        diagonal: &SparseVector<T>,
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
            SparseMatrix::<T>::new(context, &(row_height, column_width).into())?;
        let graphblas_diagonal_index = diagonal_index.as_graphblas_index()?;

        context.call_without_detailed_error_information(|| unsafe {
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
        context: &Arc<Context>,
        size: &Size,
        elements: &MatrixElementList<T>,
        reduction_operator_for_duplicates: &impl BinaryOperator<T>,
    ) -> Result<Self, SparseLinearAlgebraError>
    where
        Self: Sized;
}

// impl FromElementVector<u32> for SparseMatrix<u32> {
//     fn from_element_vector(
//         context: Arc<Context>,
//         size: Size,
//         elements: ElementVector<u32>,
//     ) -> Result<Self, GraphBlasError> {
//         let mut matrix = Self::new(context, size)?;
//         // TODO: check for duplicates
//         // TODO: check size constraints
//         matrix.context.call(|| unsafe {
//             GrB_Matrix_build_UINT32 (
//                 matrix.matrix,
//                 elements.row_index().as_ptr(),
//                 elements.column_index().as_ptr(),
//                 elements.value().as_ptr(),
//                 elements.length() as u64,
//                 GrB_FIRST_INT8,
//             )
//         })?;
//         Ok(matrix)
//     }
// }

// impl FromElementVector<u64> for SparseMatrix<u64> {
//     fn from_element_vector(
//         context: Arc<Context>,
//         size: Size,
//         elements: ElementVector<u64>,
//     ) -> Result<Self, GraphBlasError> {
//         let mut matrix = Self::new(context, size)?;
//         // TODO: check for duplicates
//         // TODO: check size constraints
//         matrix.context.call(|| unsafe {
//             GrB_Matrix_build_UINT64 (
//                 matrix.matrix,
//                 elements.row_index().as_ptr(),
//                 elements.column_index().as_ptr(),
//                 elements.value().as_ptr(),
//                 elements.length() as u64,
//                 GrB_FIRST_INT8,
//             )
//         })?;
//         Ok(matrix)
//     }
// }

macro_rules! sparse_matrix_from_element_vector {
    ($value_type:ty, $conversion_target_type: ty, $build_function:ident) => {
        impl FromMatrixElementList<$value_type> for SparseMatrix<$value_type> {
            fn from_element_list(
                context: &Arc<Context>,
                size: &Size,
                elements: &MatrixElementList<$value_type>,
                reduction_operator_for_duplicates: &impl BinaryOperator<$value_type>,
            ) -> Result<Self, SparseLinearAlgebraError> {
                // TODO: check for duplicates
                // TODO: check size constraints
                let matrix = Self::new(context, size)?;

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
                                matrix.matrix,
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

pub trait GetMatrixElement<T: ValueType> {
    fn get_element(
        &self,
        coordinate: Coordinate,
    ) -> Result<Option<MatrixElement<T>>, SparseLinearAlgebraError>;

    fn get_element_or_default(
        &self,
        coordinate: Coordinate,
    ) -> Result<MatrixElement<T>, SparseLinearAlgebraError>;
}

// impl<T: ValueType> GetMatrixElement<T> for SparseMatrix<T> {
//     fn get_element(
//         &self,
//         coordinate: Coordinate,
//     ) -> Result<MatrixElement<T>, SparseLinearAlgebraError> {
//         let value = self.get_element_value(coordinate)?;

//         Ok(MatrixElement::new(coordinate, value))
//     }
// }

macro_rules! implement_get_element {
    ($value_type:ty) => {
        impl GetMatrixElement<$value_type> for SparseMatrix<$value_type> {
            fn get_element(
                &self,
                coordinate: Coordinate,
            ) -> Result<Option<MatrixElement<$value_type>>, SparseLinearAlgebraError> {
                match self.get_element_value(&coordinate)? {
                    Some(value) => Ok(Some(MatrixElement::new(coordinate, value))),
                    None => Ok(None),
                }
            }

            fn get_element_or_default(
                &self,
                coordinate: Coordinate,
            ) -> Result<MatrixElement<$value_type>, SparseLinearAlgebraError> {
                let value = self.get_element_value_or_default(&coordinate)?;
                Ok(MatrixElement::new(coordinate, value))
            }
        }
    };
}
implement_macro_for_all_value_types!(implement_get_element);

// macro_rules! implement_get_element {
//     ($value_type:ty, $get_element_function:ident) => {
//         impl GetElement<$value_type> for SparseMatrix<$value_type> {
//             fn get_element(
//                 &self,
//                 coordinate: Coordinate,
//             ) -> Result<MatrixElement<$value_type>, SparseLinearAlgebraError> {
//                 let mut value: MaybeUninit<$value_type> = MaybeUninit::uninit();
//                 let row_index_to_get = coordinate.row_index().to_graphblas_index()?;
//                 let column_index_to_get = coordinate.column_index().to_graphblas_index()?;

//                 self.context.call(|| unsafe {
//                     $get_element_function(
//                         value.as_mut_ptr(),
//                         self.matrix,
//                         row_index_to_get,
//                         column_index_to_get,
//                     )
//                 })?;

//                 let value = unsafe { value.assume_init() };

//                 Ok(MatrixElement::new(coordinate, value))
//             }
//         }
//     };
// }

// implement_get_element!(bool, GrB_Matrix_extractElement_BOOL);
// implement_get_element!(i8, GrB_Matrix_extractElement_INT8);
// implement_get_element!(i16, GrB_Matrix_extractElement_INT16);
// implement_get_element!(i32, GrB_Matrix_extractElement_INT32);
// implement_get_element!(i64, GrB_Matrix_extractElement_INT64);
// implement_get_element!(u8, GrB_Matrix_extractElement_UINT8);
// implement_get_element!(u16, GrB_Matrix_extractElement_UINT16);
// implement_get_element!(u32, GrB_Matrix_extractElement_UINT32);
// implement_get_element!(u64, GrB_Matrix_extractElement_UINT64);
// implement_get_element!(f32, GrB_Matrix_extractElement_FP32);
// implement_get_element!(f64, GrB_Matrix_extractElement_FP64);

pub trait GetMatrixElementList<T: ValueType> {
    fn get_element_list(&self) -> Result<MatrixElementList<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_list {
    ($value_type:ty, $_graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetMatrixElementList<$value_type> for SparseMatrix<$value_type> {
            fn get_element_list(
                &self,
            ) -> Result<MatrixElementList<$value_type>, SparseLinearAlgebraError> {
                let number_of_stored_elements = self.number_of_stored_elements()?;

                let mut row_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);
                let mut column_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);
                let mut values = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                self.context.call(|| unsafe {
                    $get_element_function(
                        row_indices.as_mut_ptr(),
                        column_indices.as_mut_ptr(),
                        values.as_mut_ptr(),
                        &mut number_of_stored_and_returned_elements,
                        self.matrix,
                    )
                }, &self.matrix)?;

                let number_of_returned_elements = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if number_of_returned_elements == number_of_stored_elements {
                        row_indices.set_len(number_of_returned_elements);
                        column_indices.set_len(number_of_returned_elements);
                        values.set_len(number_of_returned_elements);
                    } else {
                        let err: SparseLinearAlgebraError = GraphBlasError::new(GraphBlasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values{}",number_of_stored_elements, number_of_returned_elements)).into();
                        return Err(err)
                    }
                };

                let row_element_indices = row_indices.into_par_iter().map(|i| ElementIndex::from_graphblas_index(i).unwrap()).collect();
                let column_element_indices = column_indices.into_par_iter().map(|i| ElementIndex::from_graphblas_index(i).unwrap()).collect();

                let values = values.to_type()?;

                let element_list = MatrixElementList::from_vectors(row_element_indices, column_element_indices, values)?;
                Ok(element_list)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_list,
    GrB_Matrix_extractTuples
);

macro_rules! implement_matrix_mask {
    ($value_type: ty) => {
        impl MatrixMask for SparseMatrix<$value_type> {
            unsafe fn graphblas_matrix(&self) -> GrB_Matrix {
                GraphblasSparseMatrixTrait::graphblas_matrix(self)
            }
        }
    };
}
implement_macro_for_all_value_types!(implement_matrix_mask);

#[cfg(test)]
mod tests {

    use super::*;
    use crate::collections::sparse_matrix::operations::{GetMatrixElementValue, SetMatrixElement};
    use crate::collections::sparse_vector::{FromVectorElementList, VectorElementList};
    use crate::context::Mode;
    use crate::error::LogicErrorType;
    use crate::operators::binary_operator::First;

    #[test]
    fn new_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height = 10;
        let target_width = 5;
        let size: Size = (target_height, target_width).into();

        let sparse_matrix = SparseMatrix::<i32>::new(&context, &size).unwrap();

        assert_eq!(target_height, sparse_matrix.row_height().unwrap());
        assert_eq!(target_width, sparse_matrix.column_width().unwrap());
        assert_eq!(0, sparse_matrix.number_of_stored_elements().unwrap());
        assert_eq!(size, sparse_matrix.size().unwrap())
    }

    #[test]
    fn clone_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height = 10;
        let target_width = 5;
        let size: Size = (target_height, target_width).into();

        let sparse_matrix = SparseMatrix::<u8>::new(&context, &size).unwrap();

        let clone_of_sparse_matrix = sparse_matrix.to_owned();

        // TODO: implement and test equality operator
        assert_eq!(target_height, clone_of_sparse_matrix.row_height().unwrap());
        assert_eq!(target_width, clone_of_sparse_matrix.column_width().unwrap());
        assert_eq!(
            0,
            clone_of_sparse_matrix.number_of_stored_elements().unwrap()
        );
        assert_eq!(size, clone_of_sparse_matrix.size().unwrap())
    }

    #[test]
    fn resize_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height = 10;
        let target_width = 5;
        let size: Size = (target_height, target_width).into();

        let mut sparse_matrix = SparseMatrix::<u8>::new(&context, &size).unwrap();

        let new_size: Size = (1, 2).into();
        sparse_matrix.resize(&new_size).unwrap();

        assert_eq!(new_size.row_height(), sparse_matrix.row_height().unwrap());
        assert_eq!(
            new_size.column_width(),
            sparse_matrix.column_width().unwrap()
        );
        assert_eq!(new_size, sparse_matrix.size().unwrap());
        // TODO: make this a meaningful test by inserting actual values
        assert_eq!(0, sparse_matrix.number_of_stored_elements().unwrap());
    }

    #[test]
    fn build_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();
        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (2, 4, 11).into(), // duplicate
                               // (10, 10, 10).into(), // out-of-bounds
        ]);
        // println!("{:?}", element_list.to_owned());

        let _matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &(3, 5).into(),
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        // println!("{:?}",matrix.get_element_list().unwrap());
        // println!("{:?}", matrix.number_of_stored_elements().unwrap());
        // println!("{:?}", matrix.number_of_stored_elements().unwrap());
        // println!("{:?}", matrix.number_of_stored_elements().unwrap());
        // assert_eq!(matrix.number_of_stored_elements().unwrap(), 3);
    }

    #[test]
    fn from_diagonal_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<isize>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (5, 5).into(),
        ]);

        let vector_length = 10;
        let vector = SparseVector::<isize>::from_element_list(
            &context,
            &vector_length,
            &element_list,
            &First::<isize>::new(),
        )
        .unwrap();

        let matrix = SparseMatrix::from_diagonal_vector(&context, &vector, &0).unwrap();
        assert_eq!(
            matrix.size().unwrap(),
            Size::new(vector_length, vector_length)
        );
        assert_eq!(
            matrix.get_element_value(&(5, 5).into()).unwrap().unwrap(),
            5
        );

        let matrix = SparseMatrix::from_diagonal_vector(&context, &vector, &2).unwrap();
        assert_eq!(
            matrix.size().unwrap(),
            Size::new(vector_length + 2, vector_length + 2)
        );
        assert_eq!(
            matrix.get_element_value(&(5, 7).into()).unwrap().unwrap(),
            5
        );

        let matrix = SparseMatrix::from_diagonal_vector(&context, &vector, &-2).unwrap();
        assert_eq!(
            matrix.size().unwrap(),
            Size::new(vector_length + 2, vector_length + 2)
        );
        println!("{}", matrix.to_owned());
        assert_eq!(
            matrix.get_element_value(&(7, 5).into()).unwrap().unwrap(),
            5
        );
    }

    #[test]
    fn set_element_in_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height = 10;
        let target_width = 5;
        let size: Size = (target_height, target_width).into();

        let mut sparse_matrix = SparseMatrix::<i32>::new(&context, &size).unwrap();

        sparse_matrix
            .set_element(MatrixElement::from_triple(1, 2, 3))
            .unwrap();

        assert_eq!(1, sparse_matrix.number_of_stored_elements().unwrap());

        sparse_matrix
            .set_element(MatrixElement::from_triple(1, 3, 3))
            .unwrap();

        assert_eq!(2, sparse_matrix.number_of_stored_elements().unwrap());

        match sparse_matrix.set_element(MatrixElement::from_triple(1, 10, 3)) {
            Err(error) => {
                match error.error_type() {
                    SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                        error_type,
                    )) => {
                        assert_eq!(error_type, GraphBlasErrorType::InvalidIndex)
                    }
                    _ => assert!(false),
                }
                // match error.error_type() {
                //     SparseLinearAlgebraErrorType::LogicErrorType(error_type) => {
                //         match error_type {
                //             LogicErrorType::GraphBlas(error_type) => {
                //                 assert_eq!(error_type, GraphBlasErrorType::InvalidIndex)
                //             }
                //             _ => assert!(false)
                //         }
                //     }
                //     _ => assert!(false)
                // }
                // assert_eq!(error.error_type(), SparseLinearAlgebraErrorType::LogicErrorType)
            }
            Ok(_) => assert!(false),
        }
    }

    #[test]
    fn remove_element_from_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height: ElementIndex = 10;
        let target_width: ElementIndex = 5;
        let size = Size::new(target_height, target_width);

        let mut sparse_matrix = SparseMatrix::<i32>::new(&context, &size).unwrap();

        sparse_matrix
            .set_element(MatrixElement::from_triple(1, 2, 3))
            .unwrap();
        sparse_matrix
            .set_element(MatrixElement::from_triple(1, 4, 4))
            .unwrap();

        sparse_matrix.drop_element(Coordinate::new(1, 2)).unwrap();

        assert_eq!(sparse_matrix.number_of_stored_elements().unwrap(), 1)
    }

    #[test]
    fn get_element_from_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height: ElementIndex = 10;
        let target_width: ElementIndex = 5;
        let size = Size::new(target_height, target_width);

        let mut sparse_matrix = SparseMatrix::<i32>::new(&context, &size).unwrap();

        let element_1 = MatrixElement::from_triple(1, 2, 1);
        let element_2 = MatrixElement::from_triple(2, 3, 2);

        sparse_matrix.set_element(element_1).unwrap();
        sparse_matrix.set_element(element_2).unwrap();

        assert_eq!(
            element_1,
            sparse_matrix
                .get_element(element_1.coordinate())
                .unwrap()
                .unwrap()
        );
        assert_eq!(
            element_2,
            sparse_matrix
                .get_element(element_2.coordinate())
                .unwrap()
                .unwrap()
        );
    }

    #[test]
    fn get_element_from_usize_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height: ElementIndex = 10;
        let target_width: ElementIndex = 5;
        let size = Size::new(target_height, target_width);

        let mut sparse_matrix = SparseMatrix::<usize>::new(&context, &size).unwrap();

        let element_1 = MatrixElement::<usize>::from_triple(1, 2, 1);
        let element_2 = MatrixElement::<usize>::from_triple(2, 3, 2);

        sparse_matrix.set_element(element_1).unwrap();
        sparse_matrix.set_element(element_2).unwrap();

        assert_eq!(
            element_1,
            sparse_matrix
                .get_element(element_1.coordinate())
                .unwrap()
                .unwrap()
        );
        assert_eq!(
            element_2,
            sparse_matrix
                .get_element(element_2.coordinate())
                .unwrap()
                .unwrap()
        );
    }

    #[test]
    fn get_element_list_from_matrix() {
        // TODO: check for a size of zero
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (2, 2, 2).into(),
            (2, 4, 10).into(),
            (2, 5, 11).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &(10, 15).into(),
            &element_list,
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

        assert_eq!(matrix.get_element_list().unwrap(), element_list);

        let empty_element_list = MatrixElementList::<u8>::new();
        let _empty_matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &(10, 15).into(),
            &empty_element_list,
            &First::<u8>::new(),
        )
        .unwrap();
        assert_eq!(
            matrix.number_of_stored_elements().unwrap(),
            element_list.length()
        );
    }

    #[test]
    fn get_test_error_reporting_while_reading_an_element() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height: ElementIndex = 10;
        let target_width: ElementIndex = 5;
        let size = Size::new(target_height, target_width);

        let mut sparse_matrix = SparseMatrix::<i32>::new(&context, &size).unwrap();

        let element_1 = MatrixElement::from_triple(1, 2, 1);
        let element_2 = MatrixElement::from_triple(20, 3, 2);

        sparse_matrix.set_element(element_1).unwrap();

        match sparse_matrix.set_element(element_2) {
            Ok(_) => assert!(false),
            Err(error) => {
                println!("{}", error.to_string());
                assert!(error
                    .to_string()
                    .contains("Row index 20 out of range; must be < 10"))
            }
        }
    }
}
