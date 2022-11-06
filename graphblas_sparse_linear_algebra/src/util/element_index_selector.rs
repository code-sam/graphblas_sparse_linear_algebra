use crate::bindings_to_graphblas_implementation::{GrB_ALL, GrB_Index};
use crate::error::SparseLinearAlgebraError;
use crate::util::{ElementIndex, IndexConversion};

#[derive(Debug)]
pub enum ElementIndexSelector<'a> {
    Index(&'a Vec<ElementIndex>),
    All,
}

pub(crate) enum ElementIndexSelectorGraphblasType {
    Index(Vec<GrB_Index>),
    All(*const GrB_Index),
}

impl<'a> ElementIndexSelector<'a> {
    pub(crate) fn to_graphblas_type(
        &self,
    ) -> Result<ElementIndexSelectorGraphblasType, SparseLinearAlgebraError> {
        match self {
            ElementIndexSelector::Index(columns) => {
                let indices: Result<Vec<GrB_Index>, SparseLinearAlgebraError> = columns
                    .into_iter()
                    .map(|index| index.to_graphblas_index())
                    .collect();

                match indices {
                    Ok(indices) => Ok(ElementIndexSelectorGraphblasType::Index(indices)),
                    Err(error) => Err(error.into()),
                }
            }
            ElementIndexSelector::All => {
                Ok(unsafe { ElementIndexSelectorGraphblasType::All(GrB_ALL) })
            }
        }
    }

    pub(crate) fn number_of_selected_elements(
        &self,
        number_elements_for_all: ElementIndex,
    ) -> Result<ElementIndex, SparseLinearAlgebraError> {
        match self {
            ElementIndexSelector::Index(indices) => Ok(indices.len()),
            ElementIndexSelector::All => Ok(number_elements_for_all),
        }
    }
}
