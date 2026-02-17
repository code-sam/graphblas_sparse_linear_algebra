use crate::index::ElementIndex;
use crate::{error::SparseLinearAlgebraError, index::ElementCount};
use crate::error::{GraphblasError, GraphblasErrorType};
use crate::error::{LogicError, LogicErrorType};

use super::{ColumnIndex, RowIndex};


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate {
    row_index: RowIndex,
    column_index: ColumnIndex,
}

pub trait GetCoordinateIndices {
    fn row_index(&self) -> RowIndex;
    fn row_index_ref(&self) -> &RowIndex;

    fn column_index(&self) -> ColumnIndex;
    fn column_index_ref(&self) -> &ColumnIndex;
}

impl GetCoordinateIndices for Coordinate {
    fn row_index(&self) -> RowIndex {
        self.row_index
    }
    fn row_index_ref(&self) -> &RowIndex {
        &self.row_index
    }

    fn column_index(&self) -> ColumnIndex {
        self.column_index
    }
    fn column_index_ref(&self) -> &ColumnIndex {
        &self.column_index
    }
}

impl Coordinate {
    pub fn new(row_index: RowIndex, column_index: RowIndex) -> Self {
        Self {
            row_index,
            column_index,
        }
    }

    pub fn from_tuple(coordinate: (RowIndex, ColumnIndex)) -> Self {
        Self::new(coordinate.0, coordinate.1)
    }
}

impl From<(RowIndex, ColumnIndex)> for Coordinate {
    fn from(coordinate: (RowIndex, ColumnIndex)) -> Self {
        Self {
            row_index: coordinate.0,
            column_index: coordinate.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateList {
    row_index: Vec<RowIndex>,
    column_index: Vec<ColumnIndex>,
}

impl CoordinateList {
    pub fn new() -> Self {
        Self {
            row_index: Vec::new(),
            column_index: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: ElementCount) -> Self {
        Self {
            row_index: Vec::with_capacity(capacity),
            column_index: Vec::with_capacity(capacity),
        }
    }

    pub fn from_coordinates(coordinates: Vec<Coordinate>) -> Self {
        let mut coordinate_list: Self = Self::with_capacity(coordinates.len());
        coordinates
            .into_iter()
            .for_each(|coordinate| coordinate_list.push_coordinate(coordinate));
        return coordinate_list;
    }

    pub fn from_vectors(
        row_index: Vec<RowIndex>,
        column_index: Vec<ColumnIndex>,
    ) -> Result<Self, SparseLinearAlgebraError> {
        #[cfg(debug_assertions)]
        if row_index.len() != column_index.len() {
            return Err(GraphblasError::new(GraphblasErrorType::DimensionMismatch,
                format!("Length of vectors must be equal: row_index.len() = {}, column_index.len() = {}", 
                row_index.len(), column_index.len()).into()).into());
        }
        Ok(Self {
            row_index,
            column_index,
        })
    }

    pub fn push_coordinate(&mut self, element: Coordinate) -> () {
        self.row_index.push(element.row_index());
        self.column_index.push(element.column_index());
    }

    pub fn append_coordinates(&mut self, elements: Vec<Coordinate>) -> () {
        let mut element_list_to_append = Self::from_coordinates(elements);
        self.row_index.append(&mut element_list_to_append.row_index);
        self.column_index
            .append(&mut element_list_to_append.column_index);
    }

    pub fn row_indices_ref(&self) -> &[RowIndex] {
        self.row_index.as_slice()
    }

    pub fn row_index(&self, index: ElementIndex) -> Result<&RowIndex, SparseLinearAlgebraError> {
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
    ) -> Result<&ColumnIndex, SparseLinearAlgebraError> {
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

    pub fn column_indices_ref(&self) -> &[ColumnIndex] {
        self.column_index.as_slice()
    }

    pub fn coordinates(&self) -> Vec<Coordinate> {
        self.row_index
            .iter()
            .zip(self.column_index.iter())
            .map(|(&row_index, &column_index)| Coordinate::new(row_index, column_index))
            .collect()
    }

    pub fn length(&self) -> ElementCount {
        self.row_index.len()
    }
}
