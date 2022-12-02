use crate::error::SparseLinearAlgebraError;
use crate::index::ElementIndex;

// // TODO: This trait is not public but will likely end up in a trait bound.
// // Can these method better be considered a non-public implementation detail not captured in a trait?
// pub(crate) trait GraphBlasCollection<T> {
//     fn graphblas_collection(&self) -> T;
//     fn graphblas_collection_ref(&self) -> &T;
//     fn graphblas_collection_mut_ref(&mut self) -> &mut T;
// }

// TODO: up what extend are these methods implementation details?
// TODO: should these methods be bundled in the same trait?
pub trait Collection {
    fn clear(&mut self) -> Result<(), SparseLinearAlgebraError>;
    fn number_of_stored_elements(&self) -> Result<ElementIndex, SparseLinearAlgebraError>;
}
