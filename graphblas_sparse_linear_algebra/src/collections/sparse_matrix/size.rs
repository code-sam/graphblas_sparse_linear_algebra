use crate::index::ElementIndex;

use super::{ColumnIndex, RowIndex};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Size {
    row_height: RowIndex,
    column_width: ColumnIndex,
}

impl Size {
    pub fn new(row_height: RowIndex, column_width: ColumnIndex) -> Self {
        Self {
            row_height,
            column_width,
        }
    }

    pub fn from_tuple(size: (RowIndex, ColumnIndex)) -> Self {
        Self::new(size.0, size.1)
    }
}

impl From<(RowIndex, ColumnIndex)> for Size {
    fn from(size: (RowIndex, ColumnIndex)) -> Self {
        Self {
            row_height: size.0,
            column_width: size.1,
        }
    }
}

pub trait GetMatrixDimensions {
    fn row_height(&self) -> RowIndex;
    fn row_height_ref(&self) -> &RowIndex;

    fn column_width(&self) -> ColumnIndex;
    fn column_width_ref(&self) -> &ColumnIndex;
}

impl GetMatrixDimensions for Size {
    fn row_height(&self) -> RowIndex {
        self.row_height
    }
    fn row_height_ref(&self) -> &RowIndex {
        &self.row_height
    }
    fn column_width(&self) -> ColumnIndex {
        self.column_width
    }
    fn column_width_ref(&self) -> &ColumnIndex {
        &self.column_width
    }
}
