use super::coordinate::Coordinate;
use super::GetCoordinateIndices;
use crate::error::{
    GraphblasError, GraphblasErrorType, LogicError, LogicErrorType, SparseLinearAlgebraError,
};
use crate::index::ElementIndex;
use crate::value_type::ValueType;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixElement<T: ValueType> {
    coordinate: Coordinate,
    value: T,
}

impl<T: ValueType> MatrixElement<T> {
    pub fn new(coordinate: Coordinate, value: T) -> Self {
        Self { coordinate, value }
    }
}

pub trait GetMatrixElementCoordinate {
    fn coordinate(&self) -> Coordinate;
    fn coordinate_ref(&self) -> &Coordinate;

    fn row_index(&self) -> ElementIndex;
    fn row_index_ref(&self) -> &ElementIndex;

    fn column_index(&self) -> ElementIndex;
    fn column_index_ref(&self) -> &ElementIndex;
}

impl<T: ValueType> GetMatrixElementCoordinate for MatrixElement<T> {
    fn coordinate(&self) -> Coordinate {
        self.coordinate
    }
    fn coordinate_ref(&self) -> &Coordinate {
        &self.coordinate
    }
    fn row_index(&self) -> ElementIndex {
        self.coordinate.row_index()
    }
    fn row_index_ref(&self) -> &ElementIndex {
        self.coordinate.row_index_ref()
    }
    fn column_index(&self) -> ElementIndex {
        self.coordinate.column_index()
    }
    fn column_index_ref(&self) -> &ElementIndex {
        self.coordinate.column_index_ref()
    }
}

pub trait GetMatrixElementValue<T> {
    fn value(&self) -> T;
    fn value_ref(&self) -> &T;
}

impl<T: ValueType + Copy> GetMatrixElementValue<T> for MatrixElement<T> {
    fn value(&self) -> T {
        self.value
    }
    fn value_ref(&self) -> &T {
        &self.value
    }
}

impl<T: ValueType + Copy> MatrixElement<T> {
    pub fn from_triple(row_index: ElementIndex, column_index: ElementIndex, value: T) -> Self {
        Self::new(Coordinate::new(row_index, column_index), value)
    }
}

impl<T: ValueType> From<(ElementIndex, ElementIndex, T)> for MatrixElement<T> {
    fn from(element: (ElementIndex, ElementIndex, T)) -> Self {
        Self {
            coordinate: Coordinate::new(element.0, element.1),
            value: element.2,
        }
    }
}

// TODO: check for uniqueness
/// Equivalent to Sparse Coordinate List (COO)
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixElementList<T: ValueType> {
    row_index: Vec<ElementIndex>,
    column_index: Vec<ElementIndex>,
    value: Vec<T>,
}

impl<T: ValueType + Clone + Copy> MatrixElementList<T> {
    pub fn new() -> Self {
        Self {
            row_index: Vec::new(),
            column_index: Vec::new(),
            value: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            row_index: Vec::with_capacity(capacity),
            column_index: Vec::with_capacity(capacity),
            value: Vec::with_capacity(capacity),
        }
    }

    pub fn from_vectors(
        row_index: Vec<ElementIndex>,
        column_index: Vec<ElementIndex>,
        value: Vec<T>,
    ) -> Result<Self, SparseLinearAlgebraError> {
        #[cfg(debug_assertions)]
        if (row_index.len() != column_index.len()) && (column_index.len() == value.len()) {
            return Err(GraphblasError::new(GraphblasErrorType::DimensionMismatch,
                format!("Length of vectors must be equal: row_index.len() = {}, column_index.len() = {}, value.len() = {}", 
                row_index.len(), column_index.len(), value.len())).into());
        }
        Ok(Self {
            row_index,
            column_index,
            value,
        })
    }

    pub fn from_element_vector(elements: Vec<MatrixElement<T>>) -> Self {
        let mut element_list: Self = Self::with_capacity(elements.len());
        elements
            .into_iter()
            .for_each(|element| element_list.push_element(element));
        return element_list;
    }

    pub fn push_element(&mut self, element: MatrixElement<T>) -> () {
        self.row_index.push(element.row_index());
        self.column_index.push(element.column_index());
        self.value.push(element.value());
    }

    pub fn append_element_vec(&mut self, elements: Vec<MatrixElement<T>>) -> () {
        let mut element_list_to_append = Self::from_element_vector(elements);
        self.row_index.append(&mut element_list_to_append.row_index);
        self.column_index
            .append(&mut element_list_to_append.column_index);
        self.value.append(&mut element_list_to_append.value);
    }

    pub fn row_indices_ref(&self) -> &[ElementIndex] {
        self.row_index.as_slice()
    }

    pub fn row_index(
        &self,
        index: ElementIndex,
    ) -> Result<&ElementIndex, SparseLinearAlgebraError> {
        #[cfg(debug_assertions)]
        if index >= self.length() {
            return Err(LogicError::new(
                LogicErrorType::IndexOutOfBounds,
                format!(
                    "index value {} larger than vector length {}",
                    index,
                    self.length()
                ),
                None,
            )
            .into());
        }
        Ok(&self.row_index[index])
    }

    pub fn column_index(
        &self,
        index: ElementIndex,
    ) -> Result<&ElementIndex, SparseLinearAlgebraError> {
        #[cfg(debug_assertions)]
        if index >= self.length() {
            return Err(LogicError::new(
                LogicErrorType::IndexOutOfBounds,
                format!(
                    "index value {} larger than vector length {}",
                    index,
                    self.length()
                ),
                None,
            )
            .into());
        }
        Ok(&self.column_index[index])
    }

    pub fn column_indices_ref(&self) -> &[ElementIndex] {
        self.column_index.as_slice()
    }

    pub fn values_ref(&self) -> &[T] {
        self.value.as_slice()
    }

    // pub fn values(&self) -> &Vec<T> {
    //     &self.value
    // }

    pub fn length(&self) -> usize {
        self.value.len()
    }
}
