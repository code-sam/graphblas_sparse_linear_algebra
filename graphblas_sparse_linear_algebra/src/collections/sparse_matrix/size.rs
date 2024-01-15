use crate::index::ElementIndex;

#[derive(Debug, Clone, PartialEq)]
pub struct Size {
    row_height: ElementIndex,
    column_width: ElementIndex,
}

impl Size {
    pub fn new(row_height: ElementIndex, column_width: ElementIndex) -> Self {
        Self {
            row_height,
            column_width,
        }
    }

    pub fn from_tuple(size: (ElementIndex, ElementIndex)) -> Self {
        Self::new(size.0, size.1)
    }
}

impl From<(ElementIndex, ElementIndex)> for Size {
    fn from(size: (ElementIndex, ElementIndex)) -> Self {
        Self {
            row_height: size.0,
            column_width: size.1,
        }
    }
}

pub trait GetMatrixDimensions {
    fn row_height_ref(&self) -> &ElementIndex;
    fn column_width_ref(&self) -> &ElementIndex;
}

impl GetMatrixDimensions for Size {
    fn row_height_ref(&self) -> &ElementIndex {
        &self.row_height
    }
    fn column_width_ref(&self) -> &ElementIndex {
        &self.column_width
    }
}
