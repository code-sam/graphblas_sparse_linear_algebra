use crate::error::{
    GraphBlasError, GraphBlasErrorType, LogicError, SparseLinearAlgebraError,
    SparseLinearAlgebraErrorType, SystemError,
};
use std::marker::PhantomData;
use std::mem::{MaybeUninit, forget};
use std::sync::Arc;

use super::coordinate::Coordinate;
use super::element::{MatrixElement, MatrixElementList};
use super::size::Size;
use crate::binary_operator::BinaryOperator;
use crate::bindings::{
    GrB_Index, GrB_Matrix, GrB_Matrix_build_BOOL, GrB_Matrix_build_FP32, GrB_Matrix_build_FP64,
    GrB_Matrix_build_INT16, GrB_Matrix_build_INT32, GrB_Matrix_build_INT64, GrB_Matrix_build_INT8,
    GrB_Matrix_build_UINT16, GrB_Matrix_build_UINT32, GrB_Matrix_build_UINT64,
    GrB_Matrix_build_UINT8, GrB_Matrix_dup, GrB_Matrix_extractElement_BOOL,
    GrB_Matrix_extractElement_FP32, GrB_Matrix_extractElement_FP64,
    GrB_Matrix_extractElement_INT16, GrB_Matrix_extractElement_INT32,
    GrB_Matrix_extractElement_INT64, GrB_Matrix_extractElement_INT8,
    GrB_Matrix_extractElement_UINT16, GrB_Matrix_extractElement_UINT32,
    GrB_Matrix_extractElement_UINT64, GrB_Matrix_extractElement_UINT8,
    GrB_Matrix_extractTuples_BOOL, GrB_Matrix_extractTuples_FP32, GrB_Matrix_extractTuples_FP64,
    GrB_Matrix_extractTuples_INT16, GrB_Matrix_extractTuples_INT32, GrB_Matrix_extractTuples_INT64,
    GrB_Matrix_extractTuples_INT8, GrB_Matrix_extractTuples_UINT16,
    GrB_Matrix_extractTuples_UINT32, GrB_Matrix_extractTuples_UINT64,
    GrB_Matrix_extractTuples_UINT8, GrB_Matrix_free, GrB_Matrix_ncols, GrB_Matrix_new,
    GrB_Matrix_nrows, GrB_Matrix_nvals, GrB_Matrix_removeElement, GrB_Matrix_resize,
    GrB_Matrix_setElement_BOOL, GrB_Matrix_setElement_FP32, GrB_Matrix_setElement_FP64,
    GrB_Matrix_setElement_INT16, GrB_Matrix_setElement_INT32, GrB_Matrix_setElement_INT64,
    GrB_Matrix_setElement_INT8, GrB_Matrix_setElement_UINT16, GrB_Matrix_setElement_UINT32,
    GrB_Matrix_setElement_UINT64, GrB_Matrix_setElement_UINT8,
};
use crate::context::Context;
use crate::util::{ElementIndex, IndexConversion};
use crate::value_type::{BuiltInValueType, CustomValueType, RegisteredCustomValueType, ValueType};

pub struct SparseMatrix<
    T: ValueType,
    const RowHeight: ElementIndex,
    const ColumnWidth: ElementIndex,
> {
    context: Arc<Context>,
    matrix: GrB_Matrix,
    value_type: PhantomData<T>,
    size: PhantomData<Size<RowHeight, ColumnWidth>>,
}

impl<
        T: ValueType + BuiltInValueType<T>,
        const RowHeight: ElementIndex,
        const ColumnWidth: ElementIndex,
    > SparseMatrix<T, RowHeight, ColumnWidth>
{
    pub fn new(context: Arc<Context>) -> Result<Self, SparseLinearAlgebraError> {
        let mut matrix: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
        let context = context.clone();

        let row_height = RowHeight.to_graphblas_index()?;
        let column_width = ColumnWidth.to_graphblas_index()?;

        context.call(|| unsafe {
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
            matrix,
            value_type: PhantomData,
            size: PhantomData,
        });
    }
}

impl<
        T: ValueType + BuiltInValueType<T>,
        const RowHeight: ElementIndex,
        const ColumnWidth: ElementIndex,
    > SparseMatrix<T, RowHeight, ColumnWidth>
{
    /// All elements of self with an index coordinate outside of the new size are dropped.
    pub fn resize<const ResizedRowHeight: ElementIndex, const ResizedColumnWidth: ElementIndex>(
        &mut self,
    ) -> Result<SparseMatrix<T, ResizedRowHeight, ResizedColumnWidth>, SparseLinearAlgebraError>
    {
        let new_row_height = ResizedRowHeight.to_graphblas_index()?;
        let new_column_width = ResizedColumnWidth.to_graphblas_index()?;

        self.context
            .call(|| unsafe { GrB_Matrix_resize(self.matrix, new_row_height, new_column_width) })?;

        let mut resized_matrix =
            SparseMatrix::<T, ResizedRowHeight, ResizedColumnWidth>::new(self.context.clone())?;
        let copy_of_resized_matrix = self.clone();
        resized_matrix.matrix = copy_of_resized_matrix.matrix;

        // TO REVIEW: does this result in a memory leak? 
        // mem::forget() prevents the regular drop() of copy_of_resized_matrix from executing. 
        // Dropping copy_of_resized_matrix frees the matrix in Graphblas
        unsafe {
            let ptr_context = Arc::into_raw(copy_of_resized_matrix.context.clone());
            Arc::decrement_strong_count(ptr_context); // decrement the clone
            Arc::decrement_strong_count(ptr_context); // decrement copy_of_resized_matrix
        }
        forget(copy_of_resized_matrix);

        Ok(
            // TODO: consider of somehow it is possible to just mutate the existing vector, instead of creating a new one.
            resized_matrix,
        )
    }
}

impl<T: ValueType, const RowHeight: ElementIndex, const ColumnWidth: ElementIndex>
    SparseMatrix<T, RowHeight, ColumnWidth>
{
    pub fn row_height(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut row_height: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context
            .call(|| unsafe { GrB_Matrix_nrows(row_height.as_mut_ptr(), self.matrix) })?;
        let row_height = unsafe { row_height.assume_init() };
        Ok(ElementIndex::from_graphblas_index(row_height)?)
    }

    pub fn column_width(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut column_width: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context
            .call(|| unsafe { GrB_Matrix_ncols(column_width.as_mut_ptr(), self.matrix) })?;
        let column_width = unsafe { column_width.assume_init() };
        Ok(ElementIndex::from_graphblas_index(column_width)?)
    }

    pub fn size(&self) -> Result<Size<RowHeight, ColumnWidth>, SparseLinearAlgebraError> {
        Ok(Size::from_tuple((self.row_height()?, self.column_width()?)))
    }

    pub fn number_of_stored_elements(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut number_of_values: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context
            .call(|| unsafe { GrB_Matrix_nvals(number_of_values.as_mut_ptr(), self.matrix) })?;
        let number_of_values = unsafe { number_of_values.assume_init() };
        Ok(ElementIndex::from_graphblas_index(number_of_values)?)
    }

    pub fn drop_element(&mut self, coordinate: Coordinate) -> Result<(), SparseLinearAlgebraError> {
        let row_index_to_delete = coordinate.row_index().to_graphblas_index()?;
        let column_index_to_delete = coordinate.column_index().to_graphblas_index()?;

        self.context.call(|| unsafe {
            GrB_Matrix_removeElement(self.matrix, row_index_to_delete, column_index_to_delete)
        })?;
        Ok(())
    }
}

impl<T: ValueType, const RowHeight: ElementIndex, const ColumnWidth: ElementIndex> Drop
    for SparseMatrix<T, RowHeight, ColumnWidth>
{
    fn drop(&mut self) -> () {
        self.context
            .call(|| unsafe { GrB_Matrix_free(&mut self.matrix.clone()) });
    }
}

impl<T: ValueType, const RowHeight: ElementIndex, const ColumnWidth: ElementIndex> Clone
    for SparseMatrix<T, RowHeight, ColumnWidth>
{
    fn clone(&self) -> Self {
        let mut matrix_copy: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
        self.context
            .call(|| unsafe { GrB_Matrix_dup(matrix_copy.as_mut_ptr(), self.matrix) })
            .unwrap();

        SparseMatrix {
            context: self.context.clone(),
            matrix: unsafe { matrix_copy.assume_init() },
            value_type: PhantomData,
            size: PhantomData,
        }
    }
}

pub trait FromElementList<
    T: ValueType,
    const RowHeight: ElementIndex,
    const ColumnWidth: ElementIndex,
>
{
    fn from_element_list(
        context: Arc<Context>,
        elements: MatrixElementList<T>,
        reduction_operator_for_duplicates: &dyn BinaryOperator<T, T, T>, // TODO: the operater could be included in the trait bound for better runtime performance
                                                                         // reduction_operator_for_duplicates: Box<dyn BinaryOperator<T, T, T>>,
    ) -> Result<SparseMatrix<T, RowHeight, ColumnWidth>, SparseLinearAlgebraError>;
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
    ($value_type:ty, $build_function:ident) => {
        impl<const RowHeight: ElementIndex, const ColumnWidth: ElementIndex>
            FromElementList<$value_type, RowHeight, ColumnWidth>
            for SparseMatrix<$value_type, RowHeight, ColumnWidth>
        {
            fn from_element_list(
                context: Arc<Context>,
                elements: MatrixElementList<$value_type>,
                reduction_operator_for_duplicates: &dyn BinaryOperator<
                    $value_type,
                    $value_type,
                    $value_type,
                >,
            ) -> Result<Self, SparseLinearAlgebraError> {
                let matrix = SparseMatrix::<$value_type, RowHeight, ColumnWidth>::new(context)?;

                let mut graphblas_row_indices = Vec::with_capacity(elements.length());
                let mut graphblas_column_indices = Vec::with_capacity(elements.length());

                for i in 0..elements.length() {
                    graphblas_row_indices.push(elements.row_index(i)?.to_graphblas_index()?);
                    graphblas_column_indices.push(elements.column_index(i)?.to_graphblas_index()?);
                }

                let number_of_elements = elements.length().to_graphblas_index()?;
                matrix.context.call(|| unsafe {
                    $build_function(
                        matrix.matrix,
                        graphblas_row_indices.as_ptr(),
                        graphblas_column_indices.as_ptr(),
                        elements.value_vec().as_ptr(),
                        number_of_elements,
                        reduction_operator_for_duplicates.graphblas_type(),
                    )
                })?;
                Ok(matrix)
            }
        }
    };
}

sparse_matrix_from_element_vector!(bool, GrB_Matrix_build_BOOL);
sparse_matrix_from_element_vector!(i8, GrB_Matrix_build_INT8);
sparse_matrix_from_element_vector!(i16, GrB_Matrix_build_INT16);
sparse_matrix_from_element_vector!(i32, GrB_Matrix_build_INT32);
sparse_matrix_from_element_vector!(i64, GrB_Matrix_build_INT64);
sparse_matrix_from_element_vector!(u8, GrB_Matrix_build_UINT8);
sparse_matrix_from_element_vector!(u16, GrB_Matrix_build_UINT16);
sparse_matrix_from_element_vector!(u32, GrB_Matrix_build_UINT32);
sparse_matrix_from_element_vector!(u64, GrB_Matrix_build_UINT64);
sparse_matrix_from_element_vector!(f32, GrB_Matrix_build_FP32);
sparse_matrix_from_element_vector!(f64, GrB_Matrix_build_FP64);

pub trait SetElement<T: ValueType, const RowHeight: ElementIndex, const ColumnWidth: ElementIndex> {
    fn set_element(&mut self, element: MatrixElement<T>) -> Result<(), SparseLinearAlgebraError>;
}

// impl SetElement<u8> for SparseMatrix<u8> {
//     fn set_element(&mut self, element: Element<u8>) -> Result<(), SparseLinearAlgebraError> {
//         self.context.call(|| unsafe {
//             GrB_Matrix_setElement_UINT8(
//                 self.matrix,
//                 element.value(),
//                 element.row_index(),
//                 element.column_index(),
//             )
//         })?;
//         Ok(())
//     }
// }

macro_rules! implement_set_element {
    ($value_type:ty, $add_element_function:ident) => {
        impl<const RowHeight: ElementIndex, const ColumnWidth: ElementIndex>
            SetElement<$value_type, RowHeight, ColumnWidth>
            for SparseMatrix<$value_type, RowHeight, ColumnWidth>
        {
            fn set_element(
                &mut self,
                element: MatrixElement<$value_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let row_index_to_set = element.row_index().to_graphblas_index()?;
                let column_index_to_set = element.column_index().to_graphblas_index()?;
                self.context.call(|| unsafe {
                    $add_element_function(
                        self.matrix,
                        element.value(),
                        row_index_to_set,
                        column_index_to_set,
                    )
                })?;
                Ok(())
            }
        }
    };
}

implement_set_element!(bool, GrB_Matrix_setElement_BOOL);
implement_set_element!(i8, GrB_Matrix_setElement_INT8);
implement_set_element!(i16, GrB_Matrix_setElement_INT16);
implement_set_element!(i32, GrB_Matrix_setElement_INT32);
implement_set_element!(i64, GrB_Matrix_setElement_INT64);
implement_set_element!(u8, GrB_Matrix_setElement_UINT8);
implement_set_element!(u16, GrB_Matrix_setElement_UINT16);
implement_set_element!(u32, GrB_Matrix_setElement_UINT32);
implement_set_element!(u64, GrB_Matrix_setElement_UINT64);
implement_set_element!(f32, GrB_Matrix_setElement_FP32);
implement_set_element!(f64, GrB_Matrix_setElement_FP64);

pub trait GetElement<T: ValueType> {
    fn get_element(
        &self,
        coordinate: Coordinate,
    ) -> Result<MatrixElement<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element {
    ($value_type:ty, $get_element_function:ident) => {
        impl<const RowHeight: ElementIndex, const ColumnWidth: ElementIndex> GetElement<$value_type>
            for SparseMatrix<$value_type, RowHeight, ColumnWidth>
        {
            fn get_element(
                &self,
                coordinate: Coordinate,
            ) -> Result<MatrixElement<$value_type>, SparseLinearAlgebraError> {
                let mut value: MaybeUninit<$value_type> = MaybeUninit::uninit();
                let row_index_to_get = coordinate.row_index().to_graphblas_index()?;
                let column_index_to_get = coordinate.column_index().to_graphblas_index()?;

                self.context.call(|| unsafe {
                    $get_element_function(
                        value.as_mut_ptr(),
                        self.matrix,
                        row_index_to_get,
                        column_index_to_get,
                    )
                })?;

                let value = unsafe { value.assume_init() };

                Ok(MatrixElement::new(coordinate, value))
            }
        }
    };
}

implement_get_element!(bool, GrB_Matrix_extractElement_BOOL);
implement_get_element!(i8, GrB_Matrix_extractElement_INT8);
implement_get_element!(i16, GrB_Matrix_extractElement_INT16);
implement_get_element!(i32, GrB_Matrix_extractElement_INT32);
implement_get_element!(i64, GrB_Matrix_extractElement_INT64);
implement_get_element!(u8, GrB_Matrix_extractElement_UINT8);
implement_get_element!(u16, GrB_Matrix_extractElement_UINT16);
implement_get_element!(u32, GrB_Matrix_extractElement_UINT32);
implement_get_element!(u64, GrB_Matrix_extractElement_UINT64);
implement_get_element!(f32, GrB_Matrix_extractElement_FP32);
implement_get_element!(f64, GrB_Matrix_extractElement_FP64);

pub trait GetElementList<T: ValueType> {
    fn get_element_list(&self) -> Result<MatrixElementList<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_list {
    ($value_type:ty, $get_element_function:ident) => {
        impl<const RowHeight: ElementIndex, const ColumnWidth: ElementIndex> GetElementList<$value_type> for SparseMatrix<$value_type, RowHeight, ColumnWidth> {
            fn get_element_list(
                &self,
            ) -> Result<MatrixElementList<$value_type>, SparseLinearAlgebraError> {
                let number_of_stored_elements = self.number_of_stored_elements()?;

                let mut row_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);
                let mut column_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);
                let mut values: Vec<$value_type> = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_returned_elements: MaybeUninit<GrB_Index> = MaybeUninit::uninit();

                self.context.call(|| unsafe {
                    $get_element_function(
                        row_indices.as_mut_ptr(),
                        column_indices.as_mut_ptr(),
                        values.as_mut_ptr(),
                        number_of_returned_elements.as_mut_ptr(),
                        self.matrix,
                    )
                })?;

                let number_of_returned_elements = unsafe {
                    number_of_returned_elements.assume_init()
                };

                let length_of_element_list = ElementIndex::from_graphblas_index(number_of_returned_elements)?;

                unsafe {
                    if length_of_element_list == number_of_stored_elements {
                        row_indices.set_len(length_of_element_list);
                        column_indices.set_len(length_of_element_list);
                        values.set_len(length_of_element_list);
                    } else {
                        let err: SparseLinearAlgebraError = GraphBlasError::new(GraphBlasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values{}",number_of_stored_elements, length_of_element_list)).into();
                        return Err(err)
                    }
                };

                let mut row_element_indices: Vec<ElementIndex> = Vec::with_capacity(length_of_element_list);
                let mut column_element_indices: Vec<ElementIndex> = Vec::with_capacity(length_of_element_list);

                for row_index in row_indices.into_iter() {
                    row_element_indices.push(ElementIndex::from_graphblas_index(row_index)?);
                }
                for column_index in column_indices.into_iter() {
                    column_element_indices.push(ElementIndex::from_graphblas_index(column_index)?);
                }

                let element_list = MatrixElementList::from_vectors(row_element_indices, column_element_indices, values)?;
                Ok(element_list)
            }
        }
    };
}

implement_get_element_list!(bool, GrB_Matrix_extractTuples_BOOL);
implement_get_element_list!(i8, GrB_Matrix_extractTuples_INT8);
implement_get_element_list!(i16, GrB_Matrix_extractTuples_INT16);
implement_get_element_list!(i32, GrB_Matrix_extractTuples_INT32);
implement_get_element_list!(i64, GrB_Matrix_extractTuples_INT64);
implement_get_element_list!(u8, GrB_Matrix_extractTuples_UINT8);
implement_get_element_list!(u16, GrB_Matrix_extractTuples_UINT16);
implement_get_element_list!(u32, GrB_Matrix_extractTuples_UINT32);
implement_get_element_list!(u64, GrB_Matrix_extractTuples_UINT64);
implement_get_element_list!(f32, GrB_Matrix_extractTuples_FP32);
implement_get_element_list!(f64, GrB_Matrix_extractTuples_FP64);

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use crate::binary_operator::First;
    use crate::context::Mode;
    use crate::error::LogicErrorType;

    #[test]
    fn new_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        const ROW_HEIGHT: ElementIndex = 10;
        const COLUMN_WIDTH: ElementIndex = 5;
        let size: Size<ROW_HEIGHT, COLUMN_WIDTH> = (ROW_HEIGHT, COLUMN_WIDTH).into();

        let sparse_matrix = SparseMatrix::<i32, ROW_HEIGHT, COLUMN_WIDTH>::new(context).unwrap();

        assert_eq!(ROW_HEIGHT, sparse_matrix.row_height().unwrap());
        assert_eq!(COLUMN_WIDTH, sparse_matrix.column_width().unwrap());
        assert_eq!(0, sparse_matrix.number_of_stored_elements().unwrap());
        assert_eq!(size, sparse_matrix.size().unwrap())
    }

    #[test]
    fn clone_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        const ROW_HEIGHT: ElementIndex = 10;
        const COLUMN_WIDTH: ElementIndex = 5;
        let size: Size<ROW_HEIGHT, COLUMN_WIDTH> = (ROW_HEIGHT, COLUMN_WIDTH).into();

        let sparse_matrix = SparseMatrix::<u8, ROW_HEIGHT, COLUMN_WIDTH>::new(context).unwrap();

        let clone_of_sparse_matrix = sparse_matrix.clone();

        // TODO: implement and test equality operator
        assert_eq!(ROW_HEIGHT, clone_of_sparse_matrix.row_height().unwrap());
        assert_eq!(COLUMN_WIDTH, clone_of_sparse_matrix.column_width().unwrap());
        assert_eq!(
            0,
            clone_of_sparse_matrix.number_of_stored_elements().unwrap()
        );
        assert_eq!(size, clone_of_sparse_matrix.size().unwrap())
    }

    #[test]
    fn resize_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        const ROW_HEIGHT: ElementIndex = 10;
        const COLUMN_WIDTH: ElementIndex = 5;
        let size: Size<ROW_HEIGHT, COLUMN_WIDTH> = (ROW_HEIGHT, COLUMN_WIDTH).into();

        let mut sparse_matrix = SparseMatrix::<u8, ROW_HEIGHT, COLUMN_WIDTH>::new(context).unwrap();

        const NEW_ROW_HEIGHT: ElementIndex = 1;
        const NEW_COLUMN_WIDTH: ElementIndex = 2;
        let new_size = Size::<NEW_ROW_HEIGHT, NEW_COLUMN_WIDTH>::new();
        let new_matrix = sparse_matrix.resize::<NEW_ROW_HEIGHT, NEW_COLUMN_WIDTH>().unwrap();

        assert_eq!(new_size.row_height(), sparse_matrix.row_height().unwrap());
        assert_eq!(
            new_size.column_width(),
            sparse_matrix.column_width().unwrap()
        );
        assert_eq!(new_size, new_matrix.size().unwrap());
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
        println!("{:?}", element_list.clone());

        let matrix = SparseMatrix::<u8, 3, 5>::from_element_list(
            context,
            element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        // println!("{:?}",matrix.get_element_list().unwrap());
        println!("{:?}", matrix.number_of_stored_elements().unwrap());
        println!("{:?}", matrix.number_of_stored_elements().unwrap());
        println!("{:?}", matrix.number_of_stored_elements().unwrap());
        assert_eq!(matrix.number_of_stored_elements().unwrap(), 3);
    }

    #[test]
    fn set_element_in_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        const ROW_HEIGHT: ElementIndex = 10;
        const COLUMN_WIDTH: ElementIndex = 5;
        let size: Size<ROW_HEIGHT, COLUMN_WIDTH> = (ROW_HEIGHT, COLUMN_WIDTH).into();

        let mut sparse_matrix =
            SparseMatrix::<i32, ROW_HEIGHT, COLUMN_WIDTH>::new(context).unwrap();

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

        const ROW_HEIGHT: ElementIndex = 10;
        const COLUMN_WIDTH: ElementIndex = 5;
        let size: Size<ROW_HEIGHT, COLUMN_WIDTH> = (ROW_HEIGHT, COLUMN_WIDTH).into();

        let mut sparse_matrix =
            SparseMatrix::<i32, ROW_HEIGHT, COLUMN_WIDTH>::new(context).unwrap();

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

        const ROW_HEIGHT: ElementIndex = 10;
        const COLUMN_WIDTH: ElementIndex = 5;
        let size: Size<ROW_HEIGHT, COLUMN_WIDTH> = (ROW_HEIGHT, COLUMN_WIDTH).into();

        let mut sparse_matrix =
            SparseMatrix::<i32, ROW_HEIGHT, COLUMN_WIDTH>::new(context).unwrap();

        let element_1 = MatrixElement::from_triple(1, 2, 1);
        let element_2 = MatrixElement::from_triple(2, 3, 2);

        sparse_matrix.set_element(element_1).unwrap();
        sparse_matrix.set_element(element_2).unwrap();

        assert_eq!(
            element_1,
            sparse_matrix.get_element(element_1.coordinate()).unwrap()
        );
        assert_eq!(
            element_2,
            sparse_matrix.get_element(element_2.coordinate()).unwrap()
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

        let matrix = SparseMatrix::<u8, 10, 15>::from_element_list(
            context.clone(),
            element_list.clone(),
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        println!("original element list: {:?}", element_list);
        println!(
            "stored element list: {:?}",
            matrix.get_element_list().unwrap()
        );
        assert_eq!(
            matrix.number_of_stored_elements().unwrap(),
            element_list.length()
        );

        assert_eq!(matrix.get_element_list().unwrap(), element_list);

        let empty_element_list = MatrixElementList::<u8>::new();
        let empty_matrix = SparseMatrix::<u8, 10, 15>::from_element_list(
            context,
            empty_element_list.clone(),
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();
        assert_eq!(
            matrix.number_of_stored_elements().unwrap(),
            element_list.length()
        );
    }
}
