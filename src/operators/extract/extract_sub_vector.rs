use std::marker::PhantomData;
use std::ptr;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator, mask::VectorMask, options::OperatorOptions,
};
use crate::sparse_vector::SparseVector;
use crate::util::{
    ElementIndex, ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion,
};
use crate::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Vector_extract,
};

#[derive(Debug, Clone)]
pub struct SubVectorExtractor<Argument, Product>
where
    Argument: ValueType,
    Product: ValueType,
{
    _argument: PhantomData<Argument>,
    _product: PhantomData<Product>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<Vector, SubVector> SubVectorExtractor<Vector, SubVector>
where
    Vector: ValueType,
    SubVector: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<SubVector, SubVector, SubVector>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _argument: PhantomData,
            _product: PhantomData,
        }
    }

    pub fn apply(
        &self,
        vector_to_extract_from: &SparseVector<Vector>,
        indices_to_extract: &ElementIndexSelector,
        sub_vector: &mut SparseVector<SubVector>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = vector_to_extract_from.context();

        let number_of_indices_to_extract: ElementIndex;
        match indices_to_extract {
            ElementIndexSelector::Index(indices) => number_of_indices_to_extract = indices.len(),
            ElementIndexSelector::All => {
                number_of_indices_to_extract = vector_to_extract_from.length()?
            }
        }
        let number_of_indices_to_extract = number_of_indices_to_extract.to_graphblas_index()?;

        let indices_to_extract = indices_to_extract.to_graphblas_type()?;

        match indices_to_extract {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(|| unsafe {
                    GrB_Vector_extract(
                        sub_vector.graphblas_vector(),
                        ptr::null_mut(),
                        self.accumulator,
                        vector_to_extract_from.graphblas_vector(),
                        index.as_ptr(),
                        number_of_indices_to_extract,
                        self.options,
                    )
                })?;
            }
            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(|| unsafe {
                    GrB_Vector_extract(
                        sub_vector.graphblas_vector(),
                        ptr::null_mut(),
                        self.accumulator,
                        vector_to_extract_from.graphblas_vector(),
                        index,
                        number_of_indices_to_extract,
                        self.options,
                    )
                })?;
            }
        }

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        vector_to_extract_from: &SparseVector<Vector>,
        indices_to_extract: &ElementIndexSelector,
        sub_vector: &mut SparseVector<SubVector>,
        mask: &VectorMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = vector_to_extract_from.context();

        let number_of_indices_to_extract: ElementIndex;
        match indices_to_extract {
            ElementIndexSelector::Index(indices) => number_of_indices_to_extract = indices.len(),
            ElementIndexSelector::All => {
                number_of_indices_to_extract = vector_to_extract_from.length()?
            }
        }
        let number_of_indices_to_extract = number_of_indices_to_extract.to_graphblas_index()?;

        let indices_to_extract = indices_to_extract.to_graphblas_type()?;

        match indices_to_extract {
            ElementIndexSelectorGraphblasType::Index(index) => {
                context.call(|| unsafe {
                    GrB_Vector_extract(
                        sub_vector.graphblas_vector(),
                        mask.graphblas_vector(),
                        self.accumulator,
                        vector_to_extract_from.graphblas_vector(),
                        index.as_ptr(),
                        number_of_indices_to_extract,
                        self.options,
                    )
                })?;
            }
            ElementIndexSelectorGraphblasType::All(index) => {
                context.call(|| unsafe {
                    GrB_Vector_extract(
                        sub_vector.graphblas_vector(),
                        mask.graphblas_vector(),
                        self.accumulator,
                        vector_to_extract_from.graphblas_vector(),
                        index,
                        number_of_indices_to_extract,
                        self.options,
                    )
                })?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::sparse_vector::{FromVectorElementList, GetVectorElementValue, VectorElementList};

    #[test]
    fn test_vector_extraction() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            (5, 5).into(),
        ]);

        let vector = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &10,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut sub_vector = SparseVector::<u8>::new(&context, &3).unwrap();

        let indices_to_extract: Vec<ElementIndex> = (0..3).collect();
        let indices_to_extract = ElementIndexSelector::Index(&indices_to_extract);

        let extractor = SubVectorExtractor::new(&OperatorOptions::new_default(), None);

        extractor
            .apply(&vector, &indices_to_extract, &mut sub_vector)
            .unwrap();

        assert_eq!(sub_vector.number_of_stored_elements().unwrap(), 2);
        assert_eq!(sub_vector.get_element_value(&2).unwrap(), 2);
    }
}
