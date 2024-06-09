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

// TODO: CoordinateList
