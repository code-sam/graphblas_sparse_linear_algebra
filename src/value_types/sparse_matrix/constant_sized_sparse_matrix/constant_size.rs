use crate::util::ElementIndex;

#[derive(Debug, Clone, PartialEq)]
pub struct Size<const R: ElementIndex, const C: ElementIndex> {
    row_height: ElementIndex,
    column_width: ElementIndex,
}

impl<const R: ElementIndex, const C: ElementIndex> Size<R, C> {
    pub fn new() -> Self {
        Self {
            row_height: R,
            column_width: C,
        }
    }

    pub fn from_tuple(size: (ElementIndex, ElementIndex)) -> Self {
        Self {
            row_height: size.0,
            column_width: size.1,
        }
    }

    pub fn row_height(&self) -> ElementIndex {
        self.row_height
    }
    pub fn column_width(&self) -> ElementIndex {
        self.column_width
    }
}

impl<const R: ElementIndex, const C: ElementIndex> From<(ElementIndex, ElementIndex)>
    for Size<R, C>
{
    fn from(size: (ElementIndex, ElementIndex)) -> Self {
        Self {
            row_height: size.0,
            column_width: size.1,
        }
    }
}
