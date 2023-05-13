use std::marker::PhantomData;
use std::ptr;

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GxB_Vector_subassign_BOOL, GxB_Vector_subassign_FP32,
    GxB_Vector_subassign_FP64, GxB_Vector_subassign_INT16, GxB_Vector_subassign_INT32,
    GxB_Vector_subassign_INT64, GxB_Vector_subassign_INT8, GxB_Vector_subassign_UINT16,
    GxB_Vector_subassign_UINT32, GxB_Vector_subassign_UINT64, GxB_Vector_subassign_UINT8,
};
use crate::collections::sparse_vector::{
    GraphblasSparseVectorTrait, SparseVector, SparseVectorTrait,
};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::index::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::operators::binary_operator::AccumulatorBinaryOperator;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_type::utilities_to_implement_traits_for_all_value_types::implement_2_type_macro_for_all_value_types_and_typed_graphblas_function_with_scalar_type_conversion;
use crate::value_type::{AsBoolean, ConvertScalar, ValueType};

// TODO: explicitly define how dupicates are handled

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<VectorToInsertInto: ValueType, ScalarToInsert: ValueType> Send
    for InsertScalarIntoSubVector<VectorToInsertInto, ScalarToInsert>
{
}
unsafe impl<VectorToInsertInto: ValueType, ScalarToInsert: ValueType> Sync
    for InsertScalarIntoSubVector<VectorToInsertInto, ScalarToInsert>
{
}

#[derive(Debug, Clone)]
pub struct InsertScalarIntoSubVector<VectorToInsertInto: ValueType, ScalarToInsert: ValueType> {
    _vector_to_insert_into: PhantomData<VectorToInsertInto>,
    _scalar_to_insert: PhantomData<ScalarToInsert>,

    accumulator: GrB_BinaryOp,
    options: GrB_Descriptor,
}

impl<VectorToInsertInto, ScalarToInsert>
    InsertScalarIntoSubVector<VectorToInsertInto, ScalarToInsert>
where
    VectorToInsertInto: ValueType,
    ScalarToInsert: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: &impl AccumulatorBinaryOperator<VectorToInsertInto>,
    ) -> Self {
        Self {
            accumulator: accumulator.accumulator_graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _vector_to_insert_into: PhantomData,
            _scalar_to_insert: PhantomData,
        }
    }
}

pub trait InsertScalarIntoSubVectorTrait<VectorToInsertInto, ScalarToInsert>
where
    VectorToInsertInto: ValueType,
    ScalarToInsert: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        vector_to_insert_into: &mut SparseVector<VectorToInsertInto>,
        indices_to_insert_into: &ElementIndexSelector,
        scalar_to_insert: &ScalarToInsert,
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        vector_to_insert_into: &mut SparseVector<VectorToInsertInto>,
        indices_to_insert_into: &ElementIndexSelector,
        scalar_to_insert: &ScalarToInsert,
        mask_for_vector_to_insert_into: &SparseVector<MaskValueType>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_insert_scalar_into_sub_vector_trait {
    (
        $_value_type_vector_to_insert_into:ty, $value_type_scalar_to_insert:ty, $graphblas_implementation_type:ty, $graphblas_insert_function:ident, $convert_to_type:ident
    ) => {
        impl<VectorToInsertInto: ValueType>
            InsertScalarIntoSubVectorTrait<VectorToInsertInto, $value_type_scalar_to_insert>
            for InsertScalarIntoSubVector<VectorToInsertInto, $value_type_scalar_to_insert>
        {
            /// replace option applies to entire matrix_to_insert_to
            fn apply(
                &self,
                vector_to_insert_into: &mut SparseVector<VectorToInsertInto>,
                indices_to_insert_into: &ElementIndexSelector,
                scalar_to_insert: &$value_type_scalar_to_insert,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = vector_to_insert_into.context();
                let scalar_to_insert = scalar_to_insert.clone().to_type()?;

                let number_of_indices_to_insert_into = indices_to_insert_into
                    .number_of_selected_elements(vector_to_insert_into.length()?)?
                    .to_graphblas_index()?;

                let indices_to_insert_into = indices_to_insert_into.to_graphblas_type()?;

                match indices_to_insert_into {
                    ElementIndexSelectorGraphblasType::Index(index) => {
                        context.call(
                            || unsafe {
                                $graphblas_insert_function(
                                    vector_to_insert_into.graphblas_vector(),
                                    ptr::null_mut(),
                                    self.accumulator,
                                    scalar_to_insert,
                                    index.as_ptr(),
                                    number_of_indices_to_insert_into,
                                    self.options,
                                )
                            },
                            unsafe { vector_to_insert_into.graphblas_vector_ref() },
                        )?;
                    }

                    ElementIndexSelectorGraphblasType::All(index) => {
                        context.call(
                            || unsafe {
                                $graphblas_insert_function(
                                    vector_to_insert_into.graphblas_vector(),
                                    ptr::null_mut(),
                                    self.accumulator,
                                    scalar_to_insert,
                                    index,
                                    number_of_indices_to_insert_into,
                                    self.options,
                                )
                            },
                            unsafe { vector_to_insert_into.graphblas_vector_ref() },
                        )?;
                    }
                }

                Ok(())
            }

            /// mask and replace option apply to entire matrix_to_insert_to
            fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
                &self,
                vector_to_insert_into: &mut SparseVector<VectorToInsertInto>,
                indices_to_insert_into: &ElementIndexSelector,
                scalar_to_insert: &$value_type_scalar_to_insert,
                mask_for_vector_to_insert_into: &SparseVector<MaskValueType>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = vector_to_insert_into.context();
                let scalar_to_insert = scalar_to_insert.clone().to_type()?;

                let number_of_indices_to_insert_into = indices_to_insert_into
                    .number_of_selected_elements(vector_to_insert_into.length()?)?
                    .to_graphblas_index()?;

                let indices_to_insert_into = indices_to_insert_into.to_graphblas_type()?;

                match indices_to_insert_into {
                    ElementIndexSelectorGraphblasType::Index(index) => {
                        context.call(
                            || unsafe {
                                $graphblas_insert_function(
                                    vector_to_insert_into.graphblas_vector(),
                                    mask_for_vector_to_insert_into.graphblas_vector(),
                                    self.accumulator,
                                    scalar_to_insert,
                                    index.as_ptr(),
                                    number_of_indices_to_insert_into,
                                    self.options,
                                )
                            },
                            unsafe { vector_to_insert_into.graphblas_vector_ref() },
                        )?;
                    }

                    ElementIndexSelectorGraphblasType::All(index) => {
                        context.call(
                            || unsafe {
                                $graphblas_insert_function(
                                    vector_to_insert_into.graphblas_vector(),
                                    mask_for_vector_to_insert_into.graphblas_vector(),
                                    self.accumulator,
                                    scalar_to_insert,
                                    index,
                                    number_of_indices_to_insert_into,
                                    self.options,
                                )
                            },
                            unsafe { vector_to_insert_into.graphblas_vector_ref() },
                        )?;
                    }
                }

                Ok(())
            }
        }
    };
}

implement_2_type_macro_for_all_value_types_and_typed_graphblas_function_with_scalar_type_conversion!(
    implement_insert_scalar_into_sub_vector_trait,
    GxB_Vector_subassign
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{Assignment, First};

    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, VectorElementList,
    };
    use crate::index::ElementIndex;

    #[test]
    fn test_insert_scalar_into_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 10).into(),
            (5, 11).into(),
        ]);

        let vector_length: usize = 10;
        let mut vector = SparseVector::<u8>::from_element_list(
            &context,
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        let mask_element_list = VectorElementList::<bool>::from_element_vector(vec![
            // (1, 1, true).into(),
            (2, true).into(),
            // (4, true).into(),
            // (5, true).into(),
        ]);
        let mask = SparseVector::<bool>::from_element_list(
            &context,
            &3,
            &mask_element_list,
            &First::<bool>::new(),
        )
        .unwrap();

        let indices_to_insert: Vec<ElementIndex> = (0..3).collect();
        let indices_to_insert = ElementIndexSelector::Index(&indices_to_insert);

        let insert_operator =
            InsertScalarIntoSubVector::new(&OperatorOptions::new_default(), &Assignment::new());

        let scalar_to_insert: u8 = 8;

        insert_operator
            .apply(&mut vector, &indices_to_insert, &scalar_to_insert)
            .unwrap();

        assert_eq!(vector.number_of_stored_elements().unwrap(), 5);
        assert_eq!(vector.get_element_value_or_default(&0).unwrap(), 8);
        assert_eq!(vector.get_element_value_or_default(&2).unwrap(), 8);
        assert_eq!(vector.get_element_value(&3).unwrap(), None);
        assert_eq!(vector.get_element_value_or_default(&5).unwrap(), 11);

        let mut vector = SparseVector::<u8>::from_element_list(
            &context,
            &vector_length,
            &element_list,
            &First::<u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply_with_mask(&mut vector, &indices_to_insert, &scalar_to_insert, &mask)
            .unwrap();

        // println!("{}", vector);

        assert_eq!(vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(vector.get_element_value(&0).unwrap(), None);
        assert_eq!(vector.get_element_value_or_default(&2).unwrap(), 8);
        assert_eq!(vector.get_element_value_or_default(&4).unwrap(), 10);
        assert_eq!(vector.get_element_value_or_default(&5).unwrap(), 11);
        assert_eq!(vector.get_element_value_or_default(&1).unwrap(), 1);
    }
}
