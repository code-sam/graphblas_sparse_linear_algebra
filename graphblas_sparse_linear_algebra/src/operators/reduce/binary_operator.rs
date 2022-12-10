use std::marker::PhantomData;
use std::ptr;

use crate::collections::collection::Collection;
use crate::collections::sparse_matrix::SparseMatrix;
use crate::collections::sparse_vector::{SparseVector, SparseVectorTrait};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};
use crate::value_types::utilities_to_implement_traits_for_all_value_types::implement_trait_for_all_value_types;
use crate::value_types::value_type::{AsBoolean, BuiltInValueType, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_reduce_BinaryOp,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
implement_trait_for_all_value_types!(Send, BinaryOperatorReducer);
implement_trait_for_all_value_types!(Sync, BinaryOperatorReducer);

#[derive(Debug, Clone)]
pub struct BinaryOperatorReducer<T: ValueType> {
    _value: PhantomData<T>,

    binary_operator: GrB_BinaryOp,
    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<T: ValueType + BuiltInValueType> BinaryOperatorReducer<T> {
    pub fn new(
        binary_operator: &dyn BinaryOperator<T, T, T>,
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<T, T, T>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            binary_operator: binary_operator.graphblas_type(),
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _value: PhantomData,
        }
    }

    pub fn to_vector(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseVector<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_reduce_BinaryOp(
                    product.graphblas_vector(),
                    ptr::null_mut(),
                    self.accumulator,
                    self.binary_operator,
                    argument.graphblas_matrix(),
                    self.options,
                )
            },
            product.graphblas_vector_ref(),
        )?;

        Ok(())
    }

    pub fn to_vector_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseVector<T>,
        mask: &SparseVector<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_Matrix_reduce_BinaryOp(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    self.accumulator,
                    self.binary_operator,
                    argument.graphblas_matrix(),
                    self.options,
                )
            },
            product.graphblas_vector_ref(),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{First, Plus};

    use crate::collections::sparse_matrix::{FromMatrixElementList, MatrixElementList, Size};
    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, VectorElementList,
    };

    #[test]
    fn test_binary_operator_reducer() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (1, 1, 1).into(),
            (1, 5, 1).into(),
            (2, 1, 2).into(),
            (4, 2, 4).into(),
            (5, 2, 5).into(),
        ]);

        let matrix_size: Size = (10, 15).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_vector =
            SparseVector::<u8>::new(&context, &matrix_size.row_height()).unwrap();

        let reducer = BinaryOperatorReducer::new(
            &Plus::<u8, u8, u8>::new(),
            &OperatorOptions::new_default(),
            None,
        );

        reducer.to_vector(&matrix, &mut product_vector).unwrap();

        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_vector.get_element_value(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);

        let mask_element_list = VectorElementList::<u8>::from_element_vector(vec![
            (1, 1).into(),
            (2, 2).into(),
            (4, 4).into(),
            // (5, 5).into(),
        ]);

        let mask = SparseVector::<u8>::from_element_list(
            &context.clone(),
            &matrix_size.row_height(),
            &mask_element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_vector =
            SparseVector::<u8>::new(&context, &matrix_size.row_height()).unwrap();

        reducer
            .to_vector_with_mask(&matrix, &mut product_vector, &mask)
            .unwrap();

        println!("{}", matrix);
        println!("{}", product_vector);

        assert_eq!(product_vector.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_vector.get_element_value(&1).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&2).unwrap(), 2);
        assert_eq!(product_vector.get_element_value(&5).unwrap(), 0);
        assert_eq!(product_vector.get_element_value(&9).unwrap(), 0);
    }
}
