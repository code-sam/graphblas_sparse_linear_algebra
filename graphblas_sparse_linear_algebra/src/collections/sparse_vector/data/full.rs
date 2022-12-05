use crate::value_types::value_type::ValueType;

pub trait FullVectorDataTrait<T: ValueType> {
    fn values(self) -> Vec<T>;
    fn values_ref(&self) -> &Vec<T>;
    fn values_mut_ref(&mut self) -> &mut Vec<T>;
    fn values_slice(&self) -> &[T];
    fn values_mut_slice(&mut self) -> &mut [T];
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct FullVectorData<T: ValueType> {
    values: Vec<T>,
}

impl<T: ValueType + Clone> FullVectorData<T> {
    pub fn new() -> Self {
        Self {
            values: Vec::<T>::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::<T>::with_capacity(capacity),
        }
    }

    pub fn from_vector(values: Vec<T>) -> Self {
        Self { values }
    }

    pub fn from_slice(values: &[T]) -> Self {
        let mut vector = Vec::with_capacity(values.len());
        vector.extend_from_slice(values);
        Self { values: vector }
    }
}

impl<T: ValueType> FullVectorDataTrait<T> for FullVectorData<T> {
    fn values(self) -> Vec<T> {
        self.values
    }
    fn values_ref(&self) -> &Vec<T> {
        &self.values
    }

    fn values_mut_ref(&mut self) -> &mut Vec<T> {
        &mut self.values
    }

    fn values_slice(&self) -> &[T] {
        self.values.as_slice()
    }

    fn values_mut_slice(&mut self) -> &mut [T] {
        self.values.as_mut_slice()
    }
}
