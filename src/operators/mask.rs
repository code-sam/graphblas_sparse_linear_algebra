use std::convert::From;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::sparse_matrix::SparseMatrix;
use crate::sparse_vector::SparseVector;
use crate::value_type::{AsBoolean, ValueType};

pub struct MatrixMask<T, MaskValueType>
where
    // T: ValueType,
    MaskValueType: AsBoolean<T>,
{
    matrix: SparseMatrix<MaskValueType>,

    _mask_value_type: PhantomData<T>,
}

impl<T: ValueType, MaskValueType: AsBoolean<T>> From<SparseMatrix<MaskValueType>>
    for MatrixMask<T, MaskValueType>
{
    fn from(matrix: SparseMatrix<MaskValueType>) -> Self {
        Self {
            matrix,
            _mask_value_type: PhantomData,
        }
    }
}

impl<T, MaskValueType> Deref for MatrixMask<T, MaskValueType>
where
    // T: ValueType,
    MaskValueType: AsBoolean<T>,
{
    type Target = SparseMatrix<MaskValueType>;

    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

pub struct VectorMask<T, MaskValueType>
where
    // T: ValueType,
    MaskValueType: AsBoolean<T>,
{
    vector: SparseVector<MaskValueType>,

    _mask_value_type: PhantomData<T>,
}

impl<T: ValueType, MaskValueType: AsBoolean<T>> From<SparseVector<MaskValueType>>
    for VectorMask<T, MaskValueType>
{
    fn from(vector: SparseVector<MaskValueType>) -> Self {
        Self {
            vector,
            _mask_value_type: PhantomData,
        }
    }
}

impl<T, MaskValueType> Deref for VectorMask<T, MaskValueType>
where
    // T: ValueType,
    MaskValueType: AsBoolean<T>,
{
    type Target = SparseVector<MaskValueType>;

    fn deref(&self) -> &Self::Target {
        &self.vector
    }
}
