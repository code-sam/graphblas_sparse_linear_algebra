use std::marker::{PhantomData, Send, Sync};
use std::mem::MaybeUninit;
use std::sync::Arc;

use suitesparse_graphblas_sys::GrB_Type;

use crate::collections::collection::Collection;
use crate::error::SparseLinearAlgebraError;
use crate::graphblas_bindings::{
    GrB_Index, GrB_Matrix, GrB_Matrix_clear, GrB_Matrix_dup, GrB_Matrix_free, GrB_Matrix_new,
    GrB_Matrix_nvals,
};
use crate::operators::mask::MatrixMask;

use super::element::MatrixElementList;
use super::size::{GetMatrixDimensions, Size};

use crate::context::GetContext;
use crate::context::{CallGraphBlasContext, Context};

use crate::collections::sparse_matrix::operations::GetSparseMatrixElementList;
use crate::collections::sparse_matrix::operations::GetSparseMatrixSize;
use crate::index::{ElementIndex, IndexConversion};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types;
use crate::value_type::ValueType;

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

pub unsafe fn new_graphblas_matrix(
    context: &Arc<Context>,
    size: &Size,
    graphblas_value_type: GrB_Type,
) -> Result<GrB_Matrix, SparseLinearAlgebraError> {
    let row_height = size.row_height_ref().to_graphblas_index()?;
    let column_width = size.column_width_ref().to_graphblas_index()?;

    let mut matrix: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();

    context.call_without_detailed_error_information(|| unsafe {
        GrB_Matrix_new(
            matrix.as_mut_ptr(),
            graphblas_value_type,
            row_height,
            column_width,
        )
    })?;

    let matrix = unsafe { matrix.assume_init() };
    return Ok(matrix);
}

impl<T: ValueType> SparseMatrix<T> {
    pub fn new(context: &Arc<Context>, size: &Size) -> Result<Self, SparseLinearAlgebraError> {
        let context = context.to_owned();

        let matrix = unsafe { new_graphblas_matrix(&context, size, T::to_graphblas_type()) }?;

        return Ok(SparseMatrix {
            context,
            matrix: matrix,
            value_type: PhantomData,
        });
    }

    pub unsafe fn from_graphblas_matrix(
        context: &Arc<Context>,
        matrix: GrB_Matrix,
    ) -> Result<SparseMatrix<T>, SparseLinearAlgebraError> {
        Ok(SparseMatrix {
            context: context.to_owned(),
            matrix,
            value_type: PhantomData,
        })
    }

    // TODO
    // fn from_matrices(matrices: Vec<SparseMatrix<T>, >) -> Result<Self, SparseLinearAlgebraError> {

    // }
}

impl<T: ValueType> GetContext for SparseMatrix<T> {
    fn context(&self) -> Arc<Context> {
        self.context.to_owned()
    }
    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl<T: ValueType> Collection for SparseMatrix<T> {
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

pub trait GetGraphblasSparseMatrix {
    unsafe fn graphblas_matrix(&self) -> GrB_Matrix;
    unsafe fn graphblas_matrix_ref(&self) -> &GrB_Matrix;
    unsafe fn graphblas_matrix_mut_ref(&mut self) -> &mut GrB_Matrix;
}

impl<T: ValueType> GetGraphblasSparseMatrix for SparseMatrix<T> {
    unsafe fn graphblas_matrix(&self) -> GrB_Matrix {
        self.matrix.to_owned()
    }

    unsafe fn graphblas_matrix_ref(&self) -> &GrB_Matrix {
        &self.matrix
    }

    unsafe fn graphblas_matrix_mut_ref(&mut self) -> &mut GrB_Matrix {
        &mut self.matrix
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
        SparseMatrix {
            context: self.context.to_owned(),
            matrix: unsafe {
                clone_graphblas_matrix(self.context_ref(), self.graphblas_matrix_ref()).unwrap()
            },
            value_type: PhantomData,
        }
    }
}

pub unsafe fn clone_graphblas_matrix(
    context: &Arc<Context>,
    matrix: &GrB_Matrix,
) -> Result<GrB_Matrix, SparseLinearAlgebraError> {
    let mut matrix_copy: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
    context
        .call(|| GrB_Matrix_dup(matrix_copy.as_mut_ptr(), *matrix), matrix)
        .unwrap();
    return Ok(matrix_copy.assume_init());
}

// TODO improve printing format
// summary data, column aligning
macro_rules! implement_display {
    ($value_type:ty) => {
        impl std::fmt::Display for SparseMatrix<$value_type> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let element_list: MatrixElementList<$value_type>;
                match self.element_list() {
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

impl<T: ValueType> MatrixMask for SparseMatrix<T> {
    unsafe fn graphblas_matrix(&self) -> GrB_Matrix {
        GetGraphblasSparseMatrix::graphblas_matrix(self)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::collections::sparse_matrix::operations::{
        DropSparseMatrixElement, FromDiagonalVector, FromMatrixElementList, GetSparseMatrixElement,
        GetSparseMatrixElementValue, GetSparseMatrixSize, ResizeSparseMatrix,
        SetSparseMatrixElement,
    };
    use crate::collections::sparse_matrix::{
        Coordinate, GetMatrixElementCoordinate, MatrixElement,
    };
    use crate::collections::sparse_vector::operations::FromVectorElementList;
    use crate::collections::sparse_vector::{SparseVector, VectorElementList};
    use crate::context::Mode;
    use crate::error::{GraphblasErrorType, LogicErrorType, SparseLinearAlgebraErrorType};
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

        assert_eq!(
            new_size.row_height_ref().to_owned(),
            sparse_matrix.row_height().unwrap()
        );
        assert_eq!(
            new_size.column_width_ref().to_owned(),
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

        let matrix = SparseMatrix::<isize>::from_diagonal_vector(&context, &vector, &0).unwrap();
        assert_eq!(
            matrix.size().unwrap(),
            Size::new(vector_length, vector_length)
        );
        assert_eq!(matrix.element_value(&5, &5).unwrap().unwrap(), 5);

        let matrix = SparseMatrix::<isize>::from_diagonal_vector(&context, &vector, &2).unwrap();
        assert_eq!(
            matrix.size().unwrap(),
            Size::new(vector_length + 2, vector_length + 2)
        );
        assert_eq!(matrix.element_value(&5, &7).unwrap().unwrap(), 5);

        let matrix = SparseMatrix::<isize>::from_diagonal_vector(&context, &vector, &-2).unwrap();
        assert_eq!(
            matrix.size().unwrap(),
            Size::new(vector_length + 2, vector_length + 2)
        );
        println!("{}", matrix.to_owned());
        assert_eq!(matrix.element_value(&7, &5).unwrap().unwrap(), 5);
    }

    #[test]
    fn set_element_in_matrix() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height = 10;
        let target_width = 5;
        let size: Size = (target_height, target_width).into();

        let mut sparse_matrix = SparseMatrix::<i32>::new(&context, &size).unwrap();

        sparse_matrix
            .set_matrix_element(&MatrixElement::from_triple(1, 2, 3))
            .unwrap();

        assert_eq!(1, sparse_matrix.number_of_stored_elements().unwrap());

        sparse_matrix
            .set_matrix_element(&MatrixElement::from_triple(1, 3, 3))
            .unwrap();

        assert_eq!(2, sparse_matrix.number_of_stored_elements().unwrap());

        match sparse_matrix.set_matrix_element(&MatrixElement::from_triple(1, 10, 3)) {
            Err(error) => {
                match error.error_type() {
                    SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                        error_type,
                    )) => {
                        assert_eq!(error_type, GraphblasErrorType::InvalidIndex)
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
            .set_matrix_element(&MatrixElement::from_triple(1, 2, 3))
            .unwrap();
        sparse_matrix
            .set_matrix_element(&MatrixElement::from_triple(1, 4, 4))
            .unwrap();

        sparse_matrix
            .drop_element_with_coordinate(&Coordinate::new(1, 2))
            .unwrap();

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

        sparse_matrix.set_matrix_element(&element_1).unwrap();
        sparse_matrix.set_matrix_element(&element_2).unwrap();

        assert_eq!(
            element_1,
            sparse_matrix
                .element(element_1.coordinate())
                .unwrap()
                .unwrap()
        );
        assert_eq!(
            element_2,
            sparse_matrix
                .element(element_2.coordinate())
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

        sparse_matrix.set_matrix_element(&element_1).unwrap();
        sparse_matrix.set_matrix_element(&element_2).unwrap();

        assert_eq!(
            element_1,
            sparse_matrix
                .element(element_1.coordinate())
                .unwrap()
                .unwrap()
        );
        assert_eq!(
            element_2,
            sparse_matrix
                .element(element_2.coordinate())
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

        assert_eq!(matrix.element_list().unwrap(), element_list);

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

        sparse_matrix.set_matrix_element(&element_1).unwrap();

        match sparse_matrix.set_matrix_element(&element_2) {
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
