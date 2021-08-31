use std::ptr;

use std::marker::PhantomData;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator, mask::VectorMask, options::OperatorOptions,
};

use crate::util::{ElementIndexSelector, ElementIndexSelectorGraphblasType, IndexConversion};
use crate::value_types::sparse_vector::SparseVector;
use crate::value_types::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GxB_Vector_subassign,
};

// TODO: explicitly define how dupicates are handled

#[derive(Debug, Clone)]
pub struct InsertVectorIntoSubVector<VectorToInsertInto: ValueType, VectorToInsert: ValueType> {
    _vector_to_insert_into: PhantomData<VectorToInsertInto>,
    _vector_to_insert: PhantomData<VectorToInsert>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<VectorToInsertInto, VectorToInsert>
    InsertVectorIntoSubVector<VectorToInsertInto, VectorToInsert>
where
    VectorToInsertInto: ValueType,
    VectorToInsert: ValueType,
{
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<
            &dyn BinaryOperator<VectorToInsert, VectorToInsertInto, VectorToInsertInto>,
        >, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _vector_to_insert_into: PhantomData,
            _vector_to_insert: PhantomData,
        }
    }
}

pub trait InsertVectorIntoSubVectorTrait<VectorToInsertInto, VectorToInsert>
where
    VectorToInsertInto: ValueType,
    VectorToInsert: ValueType,
{
    /// replace option applies to entire matrix_to_insert_to
    fn apply(
        &self,
        vector_to_insert_into: &mut SparseVector<VectorToInsertInto>,
        indices_to_insert_into: &ElementIndexSelector,
        vector_to_insert: &SparseVector<VectorToInsert>,
    ) -> Result<(), SparseLinearAlgebraError>;

    /// mask and replace option apply to entire matrix_to_insert_to
    fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        vector_to_insert_into: &mut SparseVector<VectorToInsertInto>,
        indices_to_insert_into: &ElementIndexSelector,
        vector_to_insert: &SparseVector<VectorToInsert>,
        mask_for_vector_to_insert_into: &VectorMask<MaskValueType, AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

macro_rules! implement_insert_vector_into_sub_vector_trait {
    (
        $value_type_vector_to_insert_into:ty, $value_type_vector_to_insert:ty, $graphblas_insert_function:ident
    ) => {
        impl
            InsertVectorIntoSubVectorTrait<
                $value_type_vector_to_insert_into,
                $value_type_vector_to_insert,
            >
            for InsertVectorIntoSubVector<
                $value_type_vector_to_insert_into,
                $value_type_vector_to_insert,
            >
        {
            /// replace option applies to entire matrix_to_insert_to
            fn apply(
                &self,
                vector_to_insert_into: &mut SparseVector<$value_type_vector_to_insert_into>,
                indices_to_insert_into: &ElementIndexSelector,
                vector_to_insert: &SparseVector<$value_type_vector_to_insert>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = vector_to_insert_into.context();

                let number_of_indices_to_insert_into = indices_to_insert_into
                    .number_of_selected_elements(vector_to_insert_into.length()?)?
                    .to_graphblas_index()?;

                let indices_to_insert_into = indices_to_insert_into.to_graphblas_type()?;

                match indices_to_insert_into {
                    ElementIndexSelectorGraphblasType::Index(index) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                vector_to_insert_into.graphblas_vector(),
                                ptr::null_mut(),
                                self.accumulator,
                                vector_to_insert.graphblas_vector(),
                                index.as_ptr(),
                                number_of_indices_to_insert_into,
                                self.options,
                            )
                        })?;
                    }

                    ElementIndexSelectorGraphblasType::All(index) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                vector_to_insert_into.graphblas_vector(),
                                ptr::null_mut(),
                                self.accumulator,
                                vector_to_insert.graphblas_vector(),
                                index,
                                number_of_indices_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                }

                Ok(())
            }

            /// mask and replace option apply to entire matrix_to_insert_to
            fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
                &self,
                vector_to_insert_into: &mut SparseVector<$value_type_vector_to_insert_into>,
                indices_to_insert_into: &ElementIndexSelector,
                vector_to_insert: &SparseVector<$value_type_vector_to_insert>,
                mask_for_vector_to_insert_into: &VectorMask<MaskValueType, AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = vector_to_insert_into.context();

                let number_of_indices_to_insert_into = indices_to_insert_into
                    .number_of_selected_elements(vector_to_insert_into.length()?)?
                    .to_graphblas_index()?;

                let indices_to_insert_into = indices_to_insert_into.to_graphblas_type()?;

                match indices_to_insert_into {
                    ElementIndexSelectorGraphblasType::Index(index) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                vector_to_insert_into.graphblas_vector(),
                                mask_for_vector_to_insert_into.graphblas_vector(),
                                self.accumulator,
                                vector_to_insert.graphblas_vector(),
                                index.as_ptr(),
                                number_of_indices_to_insert_into,
                                self.options,
                            )
                        })?;
                    }

                    ElementIndexSelectorGraphblasType::All(index) => {
                        context.call(|| unsafe {
                            $graphblas_insert_function(
                                vector_to_insert_into.graphblas_vector(),
                                mask_for_vector_to_insert_into.graphblas_vector(),
                                self.accumulator,
                                vector_to_insert.graphblas_vector(),
                                index,
                                number_of_indices_to_insert_into,
                                self.options,
                            )
                        })?;
                    }
                }

                Ok(())
            }
        }
    };
}

implement_insert_vector_into_sub_vector_trait!(bool, bool, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(u8, u8, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(u16, u16, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(u32, u32, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(u64, u64, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(i8, i8, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(i16, i16, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(i32, i32, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(i64, i64, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(f32, f32, GxB_Vector_subassign);
implement_insert_vector_into_sub_vector_trait!(f64, f64, GxB_Vector_subassign);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::util::ElementIndex;
    use crate::value_types::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, VectorElementList,
    };

    #[test]
    fn test_insert_vector_into_vector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 10).into(),
            (5, 12).into(),
        ]);

        let vector_length: usize = 10;
        let mut vector = SparseVector::<u8>::from_element_list(
            &context,
            &vector_length,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let element_list_to_insert = VectorElementList::<u8>::from_element_vector(vec![
            (1, 2).into(),
            (2, 3).into(),
            (4, 11).into(),
            // (5, 11).into(),
        ]);

        let vector_to_insert_length: usize = 5;
        let vector_to_insert = SparseVector::<u8>::from_element_list(
            &context,
            &vector_to_insert_length,
            &element_list_to_insert,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mask_element_list = VectorElementList::<bool>::from_element_vector(vec![
            // (1, 1, true).into(),
            (2, true).into(),
            (4, true).into(),
            // (5, true).into(),
        ]);
        let mask = SparseVector::<bool>::from_element_list(
            &context,
            &vector_to_insert_length,
            &mask_element_list,
            &First::<bool, bool, bool>::new(),
        )
        .unwrap();

        let indices_to_insert: Vec<ElementIndex> = (0..5).collect();
        let indices_to_insert = ElementIndexSelector::Index(&indices_to_insert);

        let insert_operator = InsertVectorIntoSubVector::new(&OperatorOptions::new_default(), None);

        insert_operator
            .apply(&mut vector, &indices_to_insert, &vector_to_insert)
            .unwrap();

        println!("{}", vector);

        assert_eq!(vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(vector.get_element_value(&0).unwrap(), 0);
        assert_eq!(vector.get_element_value(&2).unwrap(), 3);
        assert_eq!(vector.get_element_value(&4).unwrap(), 11);
        assert_eq!(vector.get_element_value(&5).unwrap(), 12);

        let mut vector = SparseVector::<u8>::from_element_list(
            &context,
            &vector_length,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        insert_operator
            .apply_with_mask(
                &mut vector,
                &indices_to_insert,
                &vector_to_insert,
                &mask.into(),
            )
            .unwrap();

        println!("{}", vector);

        assert_eq!(vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(vector.get_element_value(&0).unwrap(), 0);
        assert_eq!(vector.get_element_value(&2).unwrap(), 3);
        assert_eq!(vector.get_element_value(&4).unwrap(), 11);
        assert_eq!(vector.get_element_value(&5).unwrap(), 12);
        assert_eq!(vector.get_element_value(&1).unwrap(), 1);
    }
}
