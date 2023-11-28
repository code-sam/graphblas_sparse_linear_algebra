use crate::index::ElementIndex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate {
    row_index: ElementIndex,
    column_index: ElementIndex,
}

pub trait GetCoordinateIndices {
    fn row_index(&self) -> ElementIndex;
    fn row_index_ref(&self) -> &ElementIndex;

    fn column_index(&self) -> ElementIndex;
    fn column_index_ref(&self) -> &ElementIndex;
}

impl GetCoordinateIndices for Coordinate {
    fn row_index(&self) -> ElementIndex {
        self.row_index
    }
    fn row_index_ref(&self) -> &ElementIndex {
        &self.row_index
    }

    fn column_index(&self) -> ElementIndex {
        self.column_index
    }
    fn column_index_ref(&self) -> &ElementIndex {
        &self.column_index
    }
}

impl Coordinate {
    pub fn new(row_index: ElementIndex, column_index: ElementIndex) -> Self {
        Self {
            row_index,
            column_index,
        }
    }

    pub fn from_tuple(coordinate: (ElementIndex, ElementIndex)) -> Self {
        Self::new(coordinate.0, coordinate.1)
    }
}

impl From<(ElementIndex, ElementIndex)> for Coordinate {
    fn from(coordinate: (ElementIndex, ElementIndex)) -> Self {
        Self {
            row_index: coordinate.0,
            column_index: coordinate.1,
        }
    }
}

// TODO: CoordinateList
