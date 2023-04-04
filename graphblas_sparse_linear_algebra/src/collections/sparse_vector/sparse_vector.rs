use std::cmp::min;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

use once_cell::sync::Lazy;
use suitesparse_graphblas_sys::{
    GxB_Iterator, GxB_Iterator_new, GxB_Vector_Iterator_attach, GxB_Vector_Iterator_getIndex,
    GxB_Vector_Iterator_next, GxB_Vector_build_Scalar, GxB_Vector_diag, GxB_Vector_isStoredElement,
};

use super::element::{VectorElement, VectorElementList};
use crate::bindings_to_graphblas_implementation::{
    GrB_Index, GrB_Vector, GrB_Vector_build_BOOL, GrB_Vector_build_FP32, GrB_Vector_build_FP64,
    GrB_Vector_build_INT16, GrB_Vector_build_INT32, GrB_Vector_build_INT64, GrB_Vector_build_INT8,
    GrB_Vector_build_UINT16, GrB_Vector_build_UINT32, GrB_Vector_build_UINT64,
    GrB_Vector_build_UINT8, GrB_Vector_clear, GrB_Vector_dup, GrB_Vector_extractElement_BOOL,
    GrB_Vector_extractElement_FP32, GrB_Vector_extractElement_FP64,
    GrB_Vector_extractElement_INT16, GrB_Vector_extractElement_INT32,
    GrB_Vector_extractElement_INT64, GrB_Vector_extractElement_INT8,
    GrB_Vector_extractElement_UINT16, GrB_Vector_extractElement_UINT32,
    GrB_Vector_extractElement_UINT64, GrB_Vector_extractElement_UINT8,
    GrB_Vector_extractTuples_BOOL, GrB_Vector_extractTuples_FP32, GrB_Vector_extractTuples_FP64,
    GrB_Vector_extractTuples_INT16, GrB_Vector_extractTuples_INT32, GrB_Vector_extractTuples_INT64,
    GrB_Vector_extractTuples_INT8, GrB_Vector_extractTuples_UINT16,
    GrB_Vector_extractTuples_UINT32, GrB_Vector_extractTuples_UINT64,
    GrB_Vector_extractTuples_UINT8, GrB_Vector_free, GrB_Vector_new, GrB_Vector_nvals,
    GrB_Vector_removeElement, GrB_Vector_resize, GrB_Vector_setElement_BOOL,
    GrB_Vector_setElement_FP32, GrB_Vector_setElement_FP64, GrB_Vector_setElement_INT16,
    GrB_Vector_setElement_INT32, GrB_Vector_setElement_INT64, GrB_Vector_setElement_INT8,
    GrB_Vector_setElement_UINT16, GrB_Vector_setElement_UINT32, GrB_Vector_setElement_UINT64,
    GrB_Vector_setElement_UINT8, GrB_Vector_size,
};
use crate::collections::collection::Collection;
use crate::collections::sparse_matrix::{
    GraphblasSparseMatrixTrait, SparseMatrix, SparseMatrixTrait,
};
use crate::collections::sparse_scalar::{GraphblasSparseScalarTrait, SparseScalar};
use crate::context::CallGraphBlasContext;
use crate::context::{Context, ContextTrait};
use crate::error::{
    GraphBlasError, GraphBlasErrorType, LogicErrorType, SparseLinearAlgebraError,
    SparseLinearAlgebraErrorType,
};
use crate::index::{DiagonalIndex, DiagonalIndexConversion, ElementIndex, IndexConversion};
use crate::operators::binary_operator::BinaryOperator;
use crate::operators::options::OperatorOptions;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type,
    implement_macro_for_all_value_types,
};
use crate::value_type::{ConvertScalar, ConvertVector, ValueType};

static DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS: Lazy<OperatorOptions> =
    Lazy::new(|| OperatorOptions::new_default());

#[derive(Debug)]
pub struct SparseVector<T: ValueType> {
    context: Arc<Context>,
    vector: GrB_Vector,
    value_type: PhantomData<T>,
}

// Mutable access to GrB_Vector shall occur through a write lock on RwLock<GrB_Matrix>.
// Code review must consider that the correct lock is made via
// SparseMatrix::get_write_lock() and SparseMatrix::get_read_lock().
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<T: ValueType> Send for SparseVector<T> {}
unsafe impl<T: ValueType> Sync for SparseVector<T> {}

// macro_rules! new_sparse_vector {
//     ($value_type: ty, $graphblas_type: ident) => {
//         impl<T: ValueType> SparseVector<T + $value_type> {
//             pub fn new(
//                 context: &Arc<Context>,
//                 length: &ElementIndex,
//             ) -> Result<Self, SparseLinearAlgebraError> {
//                 let mut vector: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
//                 let context = context.clone();

//                 let length = length.to_graphblas_index()?;

//                 context.call_without_detailed_error_information(|| unsafe {
//                     GrB_Vector_new(vector.as_mut_ptr(), $graphblas_type, length)
//                 })?;

//                 let vector = unsafe { vector.assume_init() };
//                 return Ok(SparseVector {
//                     context,
//                     vector,
//                     value_type: PhantomData,
//                 });
//             }
//         }
//     };
// }
// implement_macro_for_all_value_types_and_graphblas_function!(new_sparse_vector, GrB);

// TODO: implement for usize and isize
impl<T: ValueType> SparseVector<T> {
    pub fn new(
        context: &Arc<Context>,
        length: &ElementIndex,
    ) -> Result<Self, SparseLinearAlgebraError> {
        let mut vector: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
        let context = context.clone();

        let length = length.to_graphblas_index()?;

        context.call_without_detailed_error_information(|| unsafe {
            GrB_Vector_new(vector.as_mut_ptr(), <T>::to_graphblas_type(), length)
        })?;

        let vector = unsafe { vector.assume_init() };
        return Ok(SparseVector {
            context,
            vector,
            value_type: PhantomData,
        });
    }

    pub fn from_sparse_scalar(
        context: &Arc<Context>,
        length: &ElementIndex,
        indices: Vec<ElementIndex>,
        value: SparseScalar<T>,
    ) -> Result<Self, SparseLinearAlgebraError> {
        let mut vector = SparseVector::<T>::new(context, length)?;
        let context = context.clone();

        let graphblas_length = indices.len().to_graphblas_index()?;

        let mut graphblas_indices = Vec::with_capacity(indices.len());
        for index in indices.into_iter() {
            graphblas_indices.push(index.to_graphblas_index()?);
        }

        context.call_without_detailed_error_information(|| unsafe {
            GxB_Vector_build_Scalar(
                // vector.as_ptr(),
                vector.graphblas_vector(),
                graphblas_indices.as_ptr(),
                value.graphblas_scalar(),
                graphblas_length,
            )
        })?;

        // let vector = unsafe { vector.assume_init() };
        // return Ok(SparseVector {
        //     context,
        //     vector,
        //     value_type: PhantomData,
        // });
        return Ok(vector);
    }

    pub fn from_sparse_matrix_diagonal(
        matrix: &SparseMatrix<T>,
        diagonal_index: &DiagonalIndex,
    ) -> Result<Self, SparseLinearAlgebraError> {
        let diagonal_length;
        if *diagonal_index > 0 {
            diagonal_length = min(
                matrix.column_width()? - diagonal_index.to_graphblas_element_index()?,
                matrix.row_height()?,
            );
        } else {
            diagonal_length = min(
                matrix.row_height()? - diagonal_index.abs().to_graphblas_element_index()?,
                matrix.column_width()?,
            );
        }

        let diagonal = SparseVector::new(matrix.context_ref(), &diagonal_length)?;
        let context = matrix.context();
        let graphblas_diagonal_index = diagonal_index.to_graphblas_index()?;

        context.call_without_detailed_error_information(|| unsafe {
            GxB_Vector_diag(
                diagonal.graphblas_vector(),
                matrix.graphblas_matrix(),
                graphblas_diagonal_index,
                DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
            )
        })?;

        return Ok(diagonal);
    }
}

// TODO: this trait is not consistent with other constructors, which do not have a trait
pub trait SparseVectorFromValue<T: ValueType> {
    fn from_value(
        context: &Arc<Context>,
        length: &ElementIndex,
        indices: Vec<ElementIndex>,
        value: T,
    ) -> Result<SparseVector<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_from_value {
    ($value_type: ty) => {
        impl SparseVectorFromValue<$value_type> for SparseVector<$value_type> {
            fn from_value(
                context: &Arc<Context>,
                length: &ElementIndex,
                indices: Vec<ElementIndex>,
                value: $value_type,
            ) -> Result<Self, SparseLinearAlgebraError> {
                let sparse_scalar: SparseScalar<$value_type> =
                    SparseScalar::<$value_type>::from_value(context, value)?;
                SparseVector::<$value_type>::from_sparse_scalar(
                    context,
                    length,
                    indices,
                    sparse_scalar,
                )
            }
        }
    };
}
implement_macro_for_all_value_types!(implement_from_value);

// impl<T: ValueType + CustomValueType> SparseVector<T> {
//     pub fn new_custom_type(
//         // context: Arc<Context>,
//         value_type: Arc<RegisteredCustomValueType<T>>,
//         length: ElementIndex,
//     ) -> Result<Self, SparseLinearAlgebraError> {
//         let mut vector: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
//         let context = value_type.context();

//         let length = length.to_graphblas_index()?;

//         context.call(|| unsafe {
//             GrB_Vector_new(vector.as_mut_ptr(), value_type.to_graphblas_type(), length)
//         })?;

//         let vector = unsafe { vector.assume_init() };
//         return Ok(SparseVector {
//             context,
//             vector,
//             value_type: PhantomData,
//         });
//     }
// }

impl<T: ValueType> ContextTrait for SparseVector<T> {
    fn context(&self) -> Arc<Context> {
        self.context.clone()
    }
    fn context_ref(&self) -> &Arc<Context> {
        &self.context
    }
}

impl<T: ValueType> Collection for SparseVector<T> {
    fn clear(&mut self) -> Result<(), SparseLinearAlgebraError> {
        self.context
            .call(|| unsafe { GrB_Vector_clear(self.vector) }, &self.vector)?;
        Ok(())
    }

    fn number_of_stored_elements(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut number_of_values: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context.call(
            || unsafe { GrB_Vector_nvals(number_of_values.as_mut_ptr(), self.vector) },
            &self.vector,
        )?;
        let number_of_values = unsafe { number_of_values.assume_init() };
        Ok(ElementIndex::from_graphblas_index(number_of_values)?)
    }
}

pub trait GraphblasSparseVectorTrait {
    unsafe fn graphblas_vector(&self) -> GrB_Vector;
    unsafe fn graphblas_vector_ref(&self) -> &GrB_Vector;
    unsafe fn graphblas_vector_mut_ref(&mut self) -> &mut GrB_Vector;
}

impl<T: ValueType> GraphblasSparseVectorTrait for SparseVector<T> {
    unsafe fn graphblas_vector(&self) -> GrB_Vector {
        self.vector
    }
    unsafe fn graphblas_vector_ref(&self) -> &GrB_Vector {
        &self.vector
    }
    unsafe fn graphblas_vector_mut_ref(&mut self) -> &mut GrB_Vector {
        &mut self.vector
    }
}

pub trait SparseVectorTrait {
    fn drop_element(
        &mut self,
        index_to_delete: ElementIndex,
    ) -> Result<(), SparseLinearAlgebraError>;
    fn is_element(&self, index: ElementIndex) -> Result<bool, SparseLinearAlgebraError>;
    fn indices(&self) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError>;
    fn length(&self) -> Result<ElementIndex, SparseLinearAlgebraError>;
    fn resize(&mut self, new_length: ElementIndex) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> SparseVectorTrait for SparseVector<T> {
    fn drop_element(
        &mut self,
        index_to_delete: ElementIndex,
    ) -> Result<(), SparseLinearAlgebraError> {
        let index_to_delete = index_to_delete.to_graphblas_index()?;

        self.context.call(
            || unsafe { GrB_Vector_removeElement(self.vector, index_to_delete) },
            &self.vector,
        )?;
        Ok(())
    }

    fn is_element(&self, index: ElementIndex) -> Result<bool, SparseLinearAlgebraError> {
        let index = index.to_graphblas_index()?;

        match self.context.call(
            || unsafe { GxB_Vector_isStoredElement(self.vector, index) },
            &self.vector,
        ) {
            Ok(_) => Ok(true),
            Err(error) => match error.error_type() {
                SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
                    GraphBlasErrorType::NoValue,
                )) => Ok(false),
                _ => Err(error),
            },
        }
    }

    fn indices(&self) -> Result<Vec<ElementIndex>, SparseLinearAlgebraError> {
        let mut iterator = MaybeUninit::uninit();
        self.context().call(
            || unsafe { GxB_Iterator_new(iterator.as_mut_ptr()) },
            &self.vector,
        )?;
        let iterator = unsafe { iterator.assume_init() };

        self.context().call(
            || unsafe {
                GxB_Vector_Iterator_attach(
                    iterator,
                    self.vector,
                    DEFAULT_GRAPHBLAS_OPERATOR_OPTIONS.to_graphblas_descriptor(),
                )
            },
            &self.vector,
        )?;

        let number_of_stored_elements = self.number_of_stored_elements()?;
        let mut indices = Vec::with_capacity(number_of_stored_elements);
        for _i in 0..number_of_stored_elements {
            println!("{:?}",_i);
            // unsafe { GxB_Vector_Iterator_next(iterator) };
            indices.push(ElementIndex::from_graphblas_index(unsafe {
                GxB_Vector_Iterator_getIndex(iterator)
            })?);
            self.context().call(
                || unsafe { GxB_Vector_Iterator_next(iterator) },
                &self.vector,
            )?;
            // println!("{:?}", unsafe{GxB_Vector_Iterator_getIndex(iterator)});
        }
        Ok(indices)
    }

    fn length(&self) -> Result<ElementIndex, SparseLinearAlgebraError> {
        let mut length: MaybeUninit<GrB_Index> = MaybeUninit::uninit();
        self.context.call(
            || unsafe { GrB_Vector_size(length.as_mut_ptr(), self.vector) },
            &self.vector,
        )?;
        let length = unsafe { length.assume_init() };
        Ok(ElementIndex::from_graphblas_index(length)?)
    }

    /// All elements of self with an index coordinate outside of the new size are dropped.
    fn resize(&mut self, new_length: ElementIndex) -> Result<(), SparseLinearAlgebraError> {
        let new_length = new_length.to_graphblas_index()?;

        self.context.call(
            || unsafe { GrB_Vector_resize(self.vector, new_length) },
            &self.vector,
        )?;
        Ok(())
    }
}

impl<T: ValueType> Drop for SparseVector<T> {
    fn drop(&mut self) -> () {
        let _ = self
            .context
            .call_without_detailed_error_information(|| unsafe {
                GrB_Vector_free(&mut self.vector.clone())
            });
    }
}

impl<T: ValueType> Clone for SparseVector<T> {
    fn clone(&self) -> Self {
        let mut vector_copy: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
        self.context
            .call(
                || unsafe { GrB_Vector_dup(vector_copy.as_mut_ptr(), self.vector) },
                &self.vector,
            )
            .unwrap();

        SparseVector {
            context: self.context.clone(),
            vector: unsafe { vector_copy.assume_init() },
            value_type: PhantomData,
        }
    }
}

// TODO: use standard GrB method
// TODO improve printing format
// summary data, column aligning
macro_rules! implement_dispay {
    ($value_type:ty) => {
        impl std::fmt::Display for SparseVector<$value_type> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let element_list: VectorElementList<$value_type>;
                match self.get_element_list() {
                    Err(_error) => return Err(std::fmt::Error),
                    Ok(list) => {
                        element_list = list;
                    }
                }

                let indices = element_list.indices_ref();
                let values = element_list.values_ref();

                writeln! {f,"Vector length: {:?}", self.length()?};
                writeln! {f,"Number of stored elements: {:?}", self.number_of_stored_elements()?};

                for element_index in 0..values.len() {
                    write!(
                        f,
                        "({}, {})\n",
                        indices[element_index], values[element_index]
                    );
                }
                return writeln!(f, "");
            }
        }
    };
}
implement_macro_for_all_value_types!(implement_dispay);

pub trait FromVectorElementList<T: ValueType> {
    fn from_element_list(
        context: &Arc<Context>,
        lenth: &ElementIndex,
        elements: &VectorElementList<T>,
        reduction_operator_for_duplicates: &dyn BinaryOperator<T, T, T, T>,
    ) -> Result<SparseVector<T>, SparseLinearAlgebraError>;
}

macro_rules! sparse_matrix_from_element_vector {
    ($value_type:ty, $graphblas_implementation_type:ty, $build_function:ident) => {
        impl FromVectorElementList<$value_type> for SparseVector<$value_type> {
            fn from_element_list(
                context: &Arc<Context>,
                length: &ElementIndex,
                elements: &VectorElementList<$value_type>,
                reduction_operator_for_duplicates: &dyn BinaryOperator<
                    $value_type,
                    $value_type,
                    $value_type,
                    $value_type,
                >,
            ) -> Result<Self, SparseLinearAlgebraError> {
                // TODO: check for duplicates
                // TODO: check size constraints
                let mut vector = Self::new(context, length)?;

                let mut graphblas_indices = Vec::with_capacity(elements.length());

                for i in 0..elements.length() {
                    graphblas_indices.push(elements.index(i)?.to_graphblas_index()?);
                }
                let number_of_elements = elements.length().to_graphblas_index()?;
                let element_values = elements.values_ref().to_owned().to_type()?;
                vector.context.call(
                    || unsafe {
                        $build_function(
                            vector.vector,
                            graphblas_indices.as_ptr(),
                            element_values.as_ptr(),
                            number_of_elements,
                            reduction_operator_for_duplicates.graphblas_type(),
                        )
                    },
                    &vector.vector,
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

pub trait SetVectorElement<T: ValueType> {
    fn set_element(&mut self, element: VectorElement<T>) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_set_element_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ident, $add_element_function:ident) => {
        impl SetVectorElement<$value_type> for SparseVector<$value_type> {
            fn set_element(
                &mut self,
                element: VectorElement<$value_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let index_to_set = element.index().to_graphblas_index()?;
                let element_value = element.value().to_type()?;
                self.context.call(
                    || unsafe { $add_element_function(self.vector, element_value, index_to_set) },
                    &self.vector,
                )?;
                Ok(())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_set_element_for_built_in_type,
    GrB_Vector_setElement
);

macro_rules! implement_set_element_for_custom_type {
    ($value_type:ty) => {
        impl SetVectorElement<$value_type> for SparseVector<$value_type> {
            fn set_element(
                &mut self,
                element: VectorElement<$value_type>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let index_to_set = element.index().to_graphblas_index()?;
                let value: *mut c_void = &mut element.value() as *mut $value_type as *mut c_void; // https://stackoverflow.com/questions/24191249/working-with-c-void-in-an-ffi
                                                                                                  // let value: *mut c_void = &mut element.value() as *mut _ as *mut c_void; // https://stackoverflow.com/questions/24191249/working-with-c-void-in-an-ffi
                self.context.call(
                    || unsafe { GrB_Vector_setElement_UDT(self.vector, value, index_to_set) },
                    &self.vector,
                )?;
                Ok(())
            }
        }
    };
}

// impl SetElement<i128> for SparseVector<i128> {
//     fn set_element(
//         &mut self,
//         element: VectorElement<i128>,
//     ) -> Result<(), SparseLinearAlgebraError> {
//         let index_to_set = element.index().to_graphblas_index()?;
//         // https://users.rust-lang.org/t/cast-c-int-to-mut-c-void/44766/3
//         let value: *mut c_void = &mut element.value() as *mut i128 as *mut c_void; // https://stackoverflow.com/questions/24191249/working-with-c-void-in-an-ffi
//         self.context.call(|| unsafe {
//             GrB_Vector_setElement_UDT(self.vector,  value, index_to_set)
//         })?;
//         Ok(())
//     }
// }

// implement_set_element_for_custom_type!(i128);
// implement_set_element_for_custom_type!(u128);

pub trait GetVectorElementValue<T: ValueType + Default> {
    fn get_element_value(
        &self,
        index: &ElementIndex,
    ) -> Result<Option<T>, SparseLinearAlgebraError>;
    fn get_element_value_or_default(
        &self,
        index: &ElementIndex,
    ) -> Result<T, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_value_for_built_in_type {
    ($value_type:ty, $graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetVectorElementValue<$value_type> for SparseVector<$value_type> {
            fn get_element_value(
                &self,
                index: &ElementIndex,
            ) -> Result<Option<$value_type>, SparseLinearAlgebraError> {
                let mut value: MaybeUninit<$graphblas_implementation_type> = MaybeUninit::uninit();
                let index_to_get = index.to_graphblas_index()?;

                let result = self.context.call(
                    || unsafe {
                        $get_element_function(value.as_mut_ptr(), self.vector, index_to_get)
                    },
                    &self.vector,
                );

                match result {
                    Ok(_) => {
                        let value = unsafe { value.assume_init() };
                        Ok(Some(value.to_type()?))
                    }
                    Err(error) => match error.error_type() {
                        SparseLinearAlgebraErrorType::LogicErrorType(
                            LogicErrorType::GraphBlas(GraphBlasErrorType::NoValue),
                        ) => Ok(None),
                        _ => Err(error),
                    },
                }
            }

            fn get_element_value_or_default(
                &self,
                index: &ElementIndex,
            ) -> Result<$value_type, SparseLinearAlgebraError> {
                Ok(self.get_element_value(index)?.unwrap_or_default())
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_value_for_built_in_type,
    GrB_Vector_extractElement
);

pub trait GetVectorElement<T: ValueType> {
    fn get_element(
        &self,
        index: ElementIndex,
    ) -> Result<Option<VectorElement<T>>, SparseLinearAlgebraError>;
    fn get_element_or_default(
        &self,
        index: ElementIndex,
    ) -> Result<VectorElement<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_for_built_in_type {
    ($value_type:ty) => {
        impl GetVectorElement<$value_type> for SparseVector<$value_type> {
            fn get_element(
                &self,
                index: ElementIndex,
            ) -> Result<Option<VectorElement<$value_type>>, SparseLinearAlgebraError> {
                match self.get_element_value(&index)? {
                    Some(value) => Ok(Some(VectorElement::new(index, value))),
                    None => Ok(None),
                }
            }

            fn get_element_or_default(
                &self,
                index: ElementIndex,
            ) -> Result<VectorElement<$value_type>, SparseLinearAlgebraError> {
                Ok(VectorElement::new(
                    index,
                    self.get_element_value_or_default(&index)?,
                ))
            }
        }
    };
}

implement_macro_for_all_value_types!(implement_get_element_for_built_in_type);

// macro_rules! implement_get_element_for_custom_type {
//     ($value_type:ty) => {
//         impl GetVectorElement<$value_type> for SparseVector<$value_type> {
//             fn get_element(
//                 &self,
//                 index: ElementIndex,
//             ) -> Result<VectorElement<$value_type>, SparseLinearAlgebraError> {
//                 let mut value: MaybeUninit<$value_type> = MaybeUninit::uninit();
//                 let pointer_to_value: *mut c_void = &mut value as *mut _ as *mut c_void; // https://stackoverflow.com/questions/24191249/working-with-c-void-in-an-ffi
//                 let index_to_get = index.to_graphblas_index()?;

//                 self.context.call(
//                     || unsafe {
//                         GrB_Vector_extractElement_UDT(pointer_to_value, self.vector, index_to_get)
//                     },
//                     &self.vector,
//                 )?;

//                 let value = unsafe { value.assume_init() };

//                 Ok(VectorElement::new(index, value))
//             }
//         }
//     };
// }

// implement_get_element_for_custom_type!(i128);
// implement_get_element_for_custom_type!(u128);

pub trait GetVectorElementList<T: ValueType> {
    fn get_element_list(&self) -> Result<VectorElementList<T>, SparseLinearAlgebraError>;
}

macro_rules! implement_get_element_list {
    ($value_type:ty, $graphblas_implementation_type:ty, $get_element_function:ident) => {
        impl GetVectorElementList<$value_type> for SparseVector<$value_type> {
            fn get_element_list(
                &self,
            ) -> Result<VectorElementList<$value_type>, SparseLinearAlgebraError> {
                let number_of_stored_elements = self.number_of_stored_elements()?;

                let mut graphblas_indices: Vec<GrB_Index> = Vec::with_capacity(number_of_stored_elements);
                let mut values: Vec<$graphblas_implementation_type> = Vec::with_capacity(number_of_stored_elements);

                let mut number_of_stored_and_returned_elements = number_of_stored_elements.as_graphblas_index()?;

                self.context.call(|| unsafe {
                    $get_element_function(
                        graphblas_indices.as_mut_ptr(),
                        values.as_mut_ptr(),
                        &mut number_of_stored_and_returned_elements,
                        self.vector)
                }, &self.vector)?;

                let length_of_element_list = ElementIndex::from_graphblas_index(number_of_stored_and_returned_elements)?;

                unsafe {
                    if length_of_element_list == number_of_stored_elements {
                        graphblas_indices.set_len(length_of_element_list);
                        values.set_len(length_of_element_list);
                    } else {
                        let err: SparseLinearAlgebraError = GraphBlasError::new(GraphBlasErrorType::IndexOutOfBounds,
                            format!("matrix.number_of_stored_elements {} unequal to length of returned values {}",number_of_stored_elements, length_of_element_list)).into();
                        return Err(err)
                    }
                };

                let mut indices: Vec<ElementIndex> = Vec::with_capacity(length_of_element_list);

                for index in graphblas_indices.into_iter() {
                    indices.push(ElementIndex::from_graphblas_index(index)?);
                }

                let values = ConvertVector::<$graphblas_implementation_type, $value_type>::to_type(values)?;
                let element_list = VectorElementList::from_vectors(indices, values)?;
                Ok(element_list)
            }
        }
    };
}

implement_1_type_macro_for_all_value_types_and_typed_graphblas_function_with_implementation_type!(
    implement_get_element_list,
    GrB_Vector_extractTuples
);

#[cfg(test)]
mod tests {

    // #[macro_use(implement_value_type_for_custom_type)]

    use super::*;
    use crate::collections::sparse_matrix::{FromMatrixElementList, MatrixElementList};
    use crate::context::Mode;
    use crate::error::LogicErrorType;
    use crate::operators::binary_operator::{First, IsGreaterThan, IsGreaterThanTyped};

    #[test]
    fn new_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let length: ElementIndex = 10;

        let sparse_vector = SparseVector::<i32>::new(&context, &length).unwrap();

        assert_eq!(length, sparse_vector.length().unwrap());
        assert_eq!(0, sparse_vector.number_of_stored_elements().unwrap());
    }

    #[test]
    fn get_indices() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let length: ElementIndex = 10;
        let value: usize = 11;
        let indices = vec![2, 3, 5];

        let sparse_vector =
            SparseVector::<usize>::from_value(&context, &length, indices.clone(), value).unwrap();
        
        assert_eq!(indices, sparse_vector.indices().unwrap());

        let indices = vec![0,1,2,3,4,5,6,7,8,9];

        let sparse_vector =
            SparseVector::<usize>::from_value(&context, &length, indices.clone(), value).unwrap();
        
        assert_eq!(indices, sparse_vector.indices().unwrap());
    }

    #[test]
    fn from_value() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let length: ElementIndex = 10;
        let value: isize = 11;
        let indices = vec![2, 3, 5];

        let sparse_vector =
            SparseVector::<isize>::from_value(&context, &length, indices.clone(), value).unwrap();

        assert_eq!(length, sparse_vector.length().unwrap());
        assert_eq!(
            indices.len(),
            sparse_vector.number_of_stored_elements().unwrap()
        );
        for index in indices {
            assert_eq!(
                sparse_vector.get_element_value_or_default(&index).unwrap(),
                value
            );
        }
    }

    #[test]
    fn from_sparse_diagonal() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 0).into(),
            (0, 1, 1).into(),
            (0, 2, 2).into(),
            (2, 2, 4).into(),
            (2, 4, 6).into(),
            (2, 5, 10).into(),
            (3, 1, 4).into(),
        ]);

        let matrix = SparseMatrix::<u8>::from_element_list(
            &context,
            &(10, 15).into(),
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let diagonal = SparseVector::from_sparse_matrix_diagonal(&matrix, &0).unwrap();
        assert_eq!(diagonal.length().unwrap(), 10);
        assert_eq!(diagonal.number_of_stored_elements().unwrap(), 2);
        assert_eq!(diagonal.get_element_value_or_default(&0).unwrap(), 0);
        assert_eq!(diagonal.get_element_value_or_default(&2).unwrap(), 4);

        let diagonal = SparseVector::from_sparse_matrix_diagonal(&matrix, &2).unwrap();
        assert_eq!(diagonal.length().unwrap(), 10);
        assert_eq!(diagonal.number_of_stored_elements().unwrap(), 2);
        assert_eq!(diagonal.get_element_value_or_default(&0).unwrap(), 2);
        assert_eq!(diagonal.get_element_value_or_default(&2).unwrap(), 6);

        let diagonal = SparseVector::from_sparse_matrix_diagonal(&matrix, &-2).unwrap();
        assert_eq!(diagonal.length().unwrap(), 8);
        assert_eq!(diagonal.number_of_stored_elements().unwrap(), 1);
        assert_eq!(diagonal.get_element_value_or_default(&1).unwrap(), 4);
    }

    #[test]
    fn clone_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let length: ElementIndex = 10;

        let sparse_vector = SparseVector::<f32>::new(&context, &length).unwrap();

        let clone_of_sparse_vector = sparse_vector.clone();

        // TODO: implement and test equality operator
        assert_eq!(length, sparse_vector.length().unwrap());
        assert_eq!(
            0,
            clone_of_sparse_vector.number_of_stored_elements().unwrap()
        );
    }

    #[test]
    fn resize_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let length: ElementIndex = 10;

        let mut sparse_vector = SparseVector::<i32>::new(&context, &length).unwrap();

        let new_length: ElementIndex = 5;
        sparse_vector.resize(new_length.clone()).unwrap();

        assert_eq!(new_length, sparse_vector.length().unwrap());

        // TODO: make this a meaningful test by inserting actual values
        assert_eq!(0, sparse_vector.number_of_stored_elements().unwrap());
    }

    #[test]
    fn build_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();
        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (4, 2).into(),
            (2, 10).into(),
            (2, 11).into(), // duplicate
                            // (10, 10, 10).into(), // out-of-bounds
        ]);
        // println!("{:?}", element_list.clone());

        let vector = SparseVector::<u8>::from_element_list(
            &context,
            &10,
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        // println!("{:?}",matrix.get_element_list().unwrap());
        // println!("{:?}", vector.number_of_stored_elements().unwrap());
        // println!("{:?}", vector.number_of_stored_elements().unwrap());
        // println!("{:?}", vector.number_of_stored_elements().unwrap());
        assert_eq!(vector.number_of_stored_elements().unwrap(), 3);
    }

    #[test]
    fn set_element_in_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let length: ElementIndex = 10;

        let mut sparse_vector = SparseVector::<i32>::new(&context, &length).unwrap();

        sparse_vector
            .set_element(VectorElement::from_pair(1, 2))
            .unwrap();

        assert_eq!(1, sparse_vector.number_of_stored_elements().unwrap());

        sparse_vector
            .set_element(VectorElement::from_pair(3, 3))
            .unwrap();

        assert_eq!(2, sparse_vector.number_of_stored_elements().unwrap());

        match sparse_vector.set_element(VectorElement::from_pair(15, 3)) {
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

    // #[test]
    // fn set_element_in_vector_custom_type() {
    //     let context = Context::init_ready(Mode::NonBlocking).unwrap();

    //     #[repr(C)]
    //     #[derive(Clone, Debug, PartialEq)]
    //     struct CustomType {
    //         prop_1: u64,
    //         prop_2: i16,
    //     }
    //     impl CustomType {
    //         pub fn new(prop_1: u64, prop_2: i16) -> Self {
    //             CustomType {prop_1, prop_2}
    //         }
    //     }

    //     crate::implement_value_type_for_custom_type!(CustomType);
    //     implement_set_element_for_custom_type!(CustomType);

    //     let value_type_i128 = CustomType::register(context).unwrap();
    //     println!("size in Graphblas: {}",value_type_i128.size_in_graphblas().unwrap());

    //     let length: ElementIndex = 10;

    //     let mut sparse_vector = SparseVector::<CustomType>::new(value_type_i128, length.clone()).unwrap();

    //     sparse_vector
    //         .set_element(VectorElement::from_pair(1, CustomType::new(2,2)))
    //         .unwrap();

    //     assert_eq!(1, sparse_vector.number_of_stored_elements().unwrap());

    //     sparse_vector
    //         .set_element(VectorElement::from_pair(3, CustomType::new(3,3)))
    //         .unwrap();

    //     assert_eq!(2, sparse_vector.number_of_stored_elements().unwrap());

    //     match sparse_vector.set_element(VectorElement::from_pair(15, CustomType::new(4,2))) {
    //         Err(error) => {
    //             match error.error_type() {
    //                 SparseLinearAlgebraErrorType::LogicErrorType(LogicErrorType::GraphBlas(
    //                     error_type,
    //                 )) => {
    //                     assert_eq!(error_type, GraphBlasErrorType::InvalidIndex)
    //                 }
    //                 _ => assert!(false),
    //             }
    //             // match error.error_type() {
    //             //     SparseLinearAlgebraErrorType::LogicErrorType(error_type) => {
    //             //         match error_type {
    //             //             LogicErrorType::GraphBlas(error_type) => {
    //             //                 assert_eq!(error_type, GraphBlasErrorType::InvalidIndex)
    //             //             }
    //             //             _ => assert!(false)
    //             //         }
    //             //     }
    //             //     _ => assert!(false)
    //             // }
    //             // assert_eq!(error.error_type(), SparseLinearAlgebraErrorType::LogicErrorType)
    //         }
    //         Ok(_) => assert!(false),
    //     }
    // }

    #[test]
    fn remove_element_from_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let length: ElementIndex = 10;

        let mut sparse_vector = SparseVector::<i64>::new(&context, &length).unwrap();

        sparse_vector
            .set_element(VectorElement::from_pair(2, 3))
            .unwrap();
        sparse_vector
            .set_element(VectorElement::from_pair(4, 4))
            .unwrap();

        sparse_vector.drop_element(2).unwrap();

        assert_eq!(sparse_vector.number_of_stored_elements().unwrap(), 1)
    }

    #[test]
    fn get_element_from_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let length: ElementIndex = 10;

        let mut sparse_vector = SparseVector::<u8>::new(&context, &length).unwrap();

        let element_1 = VectorElement::from_pair(1, 2);
        let element_2 = VectorElement::from_pair(2, 3);

        sparse_vector.set_element(element_1).unwrap();
        sparse_vector.set_element(element_2).unwrap();

        assert_eq!(
            element_1,
            sparse_vector
                .get_element_or_default(element_1.index())
                .unwrap()
        );
        assert_eq!(
            element_2,
            sparse_vector
                .get_element_or_default(element_2.index())
                .unwrap()
        );
    }

    // #[test]
    // fn get_element_from_vector_custom_type() {
    // let context = Context::init_ready(Mode::NonBlocking).unwrap();
    // let custom_u128 = u128::register(context).unwrap();

    // let length: ElementIndex = 10;

    // let mut sparse_vector = SparseVector::<u128>::new(custom_u128, length.clone()).unwrap();

    // let element_1 = VectorElement::from_pair(1, 2);
    // let element_2 = VectorElement::from_pair(2, 3);

    // sparse_vector.set_element(element_1).unwrap();
    // sparse_vector.set_element(element_2).unwrap();

    // assert_eq!(
    //     element_1,
    //     sparse_vector.get_element(element_1.index()).unwrap()
    // );
    // assert_eq!(
    //     element_2,
    //     sparse_vector.get_element(element_2.index()).unwrap()
    // );
    // }

    #[test]
    fn get_element_list_from_matrix() {
        // TODO: check for a size of zero
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &10,
            &element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        // println!("original element list: {:?}", element_list);
        // println!(
        //     "stored element list: {:?}",
        //     vector.get_element_list().unwrap()
        // );
        assert_eq!(
            vector.number_of_stored_elements().unwrap(),
            element_list.length()
        );

        assert_eq!(vector.get_element_list().unwrap(), element_list);

        let empty_element_list = VectorElementList::<u8>::new();
        let _empty_matrix = SparseVector::<u8>::from_element_list(
            &context,
            &10,
            &empty_element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();
        assert_eq!(
            vector.number_of_stored_elements().unwrap(),
            element_list.length()
        );
    }

}
