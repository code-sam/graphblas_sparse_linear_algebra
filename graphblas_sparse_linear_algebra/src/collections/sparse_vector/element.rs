use crate::error::{
    GraphBlasError, GraphBlasErrorType, LogicError, LogicErrorType, SparseLinearAlgebraError,
};
use crate::index::ElementIndex;
use crate::value_type::ValueType;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorElement<T: ValueType> {
    index: ElementIndex,
    value: T,
}

impl<T: ValueType> VectorElement<T> {
    pub fn new(index: ElementIndex, value: T) -> Self {
        Self { index, value }
    }
}

impl<T: ValueType + Clone> VectorElement<T> {
    pub fn index(&self) -> ElementIndex {
        self.index.clone()
    }
    pub fn value(&self) -> T {
        self.value.clone()
    }

    pub fn from_pair(index: ElementIndex, value: T) -> Self {
        Self::new(index, value)
    }
}

impl<T: ValueType> From<(ElementIndex, T)> for VectorElement<T> {
    fn from(element: (ElementIndex, T)) -> Self {
        Self {
            index: element.0,
            value: element.1,
        }
    }
}

// TODO: check for uniqueness
/// Equivalent to Sparse Coordinate List (COO)
#[derive(Debug, Clone, PartialEq)]
pub struct VectorElementList<T: ValueType> {
    // elements: Vec<Element<T>>,
    index: Vec<ElementIndex>,
    value: Vec<T>,
}

impl<T: ValueType + Clone + Copy> VectorElementList<T> {
    pub fn new() -> Self {
        Self {
            index: Vec::new(),
            value: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            index: Vec::with_capacity(capacity),
            value: Vec::with_capacity(capacity),
        }
    }

    pub fn from_vectors(
        index: Vec<ElementIndex>,
        value: Vec<T>,
    ) -> Result<Self, SparseLinearAlgebraError> {
        #[cfg(debug_assertions)]
        if index.len() != value.len() {
            return Err(GraphBlasError::new(
                GraphBlasErrorType::DimensionMismatch,
                format!(
                    "Length of vectors must be equal: index.len() = {}, value.len() = {}",
                    index.len(),
                    value.len()
                ),
            )
            .into());
        }
        Ok(Self { index, value })
    }

    pub fn from_element_vector(elements: Vec<VectorElement<T>>) -> Self {
        let mut element_list: Self = Self::with_capacity(elements.len());
        elements
            .into_iter()
            .for_each(|element| element_list.push_element(element));
        return element_list;
    }

    pub fn push_element(&mut self, element: VectorElement<T>) -> () {
        self.index.push(element.index());
        self.value.push(element.value());
    }

    pub fn append_element_vec(&mut self, elements: Vec<VectorElement<T>>) -> () {
        let mut element_list_to_append = Self::from_element_vector(elements);
        self.index.append(&mut element_list_to_append.index);
        self.value.append(&mut element_list_to_append.value);
    }

    pub fn indices_ref(&self) -> &[ElementIndex] {
        self.index.as_slice()
    }

    pub(crate) fn index(&self, index: ElementIndex) -> Result<&ElementIndex, LogicError> {
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
            ));
        }
        Ok(&self.index[index])
    }

    pub fn values_ref(&self) -> &[T] {
        self.value.as_slice()
    }

    // pub fn as_element_vec(&self) -> &Vec<Element<T>> {
    //     &self.elements
    // }

    // pub fn as_element_vec_mut(&mut self) -> &mut Vec<Element<T>> {
    //     &mut self.elements
    // }

    pub fn length(&self) -> usize {
        self.value.len()
    }
}

// impl<T: ValueType> From<(Vec<Index>, Vec<Index>, Vec<T>)> for ElementVector<T> {
//     fn from(elements: (Vec<Index>, Vec<Index>, Vec<T>)) -> Self {
//         ElementVector {
//             row_index: elements.0,
//             column_index: elements.1,
//             value: elements.2,
//         }
//     }
// }

// impl<T: ValueType + Clone> From<(&[Index], &[Index], &[T])> for ElementVector<T> {
//     fn from(elements: (&[Index], &[Index], &[T])) -> Self {
//         ElementVector {
//             row_index: elements.0.to_vec(),
//             column_index: elements.1.to_vec(),
//             value: elements.2.to_vec(),
//         }
//     }
// }
