use std::convert::TryInto;
use std::marker::PhantomData;
use std::ptr;

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_reduce_BOOL, GrB_Matrix_reduce_FP32,
    GrB_Matrix_reduce_FP64, GrB_Matrix_reduce_INT16, GrB_Matrix_reduce_INT32,
    GrB_Matrix_reduce_INT64, GrB_Matrix_reduce_INT8, GrB_Matrix_reduce_Monoid,
    GrB_Matrix_reduce_UINT16, GrB_Matrix_reduce_UINT32, GrB_Matrix_reduce_UINT64,
    GrB_Matrix_reduce_UINT8, GrB_Monoid, GrB_Vector_reduce_BOOL, GrB_Vector_reduce_FP32,
    GrB_Vector_reduce_FP64, GrB_Vector_reduce_INT16, GrB_Vector_reduce_INT32,
    GrB_Vector_reduce_INT64, GrB_Vector_reduce_INT8, GrB_Vector_reduce_UINT16,
    GrB_Vector_reduce_UINT32, GrB_Vector_reduce_UINT64, GrB_Vector_reduce_UINT8,
};
use crate::error::SparseLinearAlgebraError;
use crate::operators::{binary_operator::BinaryOperator, monoid::Monoid, options::OperatorOptions};
use crate::value_types::sparse_matrix::SparseMatrix;
use crate::value_types::sparse_vector::SparseVector;
use crate::value_types::utilities_to_implement_traits_for_all_value_types::{
    convert_mut_scalar_to_type, identity_conversion,
    implement_macro_for_all_value_types_and_2_typed_graphblas_functions_with_mutable_scalar_type_conversion,
    implement_trait_for_all_value_types,
};
use crate::value_types::value_type::{AsBoolean, ValueType};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
implement_trait_for_all_value_types!(Send, MonoidReducer);
implement_trait_for_all_value_types!(Sync, MonoidReducer);

#[derive(Debug, Clone)]
pub struct MonoidReducer<T: ValueType> {
    _value: PhantomData<T>,

    monoid: GrB_Monoid,
    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

pub trait MonoidScalarReducer<T>
where
    T: ValueType,
{
    fn matrix_to_scalar(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn vector_to_scalar(
        &self,
        argument: &SparseVector<T>,
        product: &mut T,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<T: ValueType> MonoidReducer<T> {
    pub fn new(
        monoid: &dyn Monoid<T>,
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<T, T, T>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            monoid: monoid.graphblas_type(),
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

        context.call(|| unsafe {
            GrB_Matrix_reduce_Monoid(
                product.graphblas_vector(),
                ptr::null_mut(),
                self.accumulator,
                self.monoid,
                argument.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }

    pub fn to_vector_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseVector<T>,
        mask: &SparseVector<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_reduce_Monoid(
                product.graphblas_vector(),
                mask.graphblas_vector(),
                self.accumulator,
                self.monoid,
                argument.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }
}

macro_rules! implement_monoid_reducer {
    ($value_type:ty, $graphblas_implementation_type:ty, $matrix_reducer_operator:ident, $vector_reducer_operator:ident, $convert_to_type:ident) => {
        impl MonoidScalarReducer<$value_type> for MonoidReducer<$value_type> {
            fn matrix_to_scalar(
                &self,
                argument: &SparseMatrix<$value_type>,
                product: &mut $value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();
                let mut tmp_product = product.clone();
                $convert_to_type!(tmp_product, $graphblas_implementation_type);

                context.call(|| unsafe {
                    $matrix_reducer_operator(
                        &mut tmp_product,
                        self.accumulator,
                        self.monoid,
                        argument.graphblas_matrix(),
                        self.options,
                    )
                })?;

                $convert_to_type!(tmp_product, $value_type);
                *product = tmp_product;
                Ok(())
            }

            fn vector_to_scalar(
                &self,
                argument: &SparseVector<$value_type>,
                product: &mut $value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = argument.context();
                let mut tmp_product = product.clone();
                $convert_to_type!(tmp_product, $graphblas_implementation_type);

                context.call(|| unsafe {
                    $vector_reducer_operator(
                        &mut tmp_product,
                        self.accumulator,
                        self.monoid,
                        argument.graphblas_vector(),
                        self.options,
                    )
                })?;

                $convert_to_type!(tmp_product, $value_type);
                *product = tmp_product;
                Ok(())
            }
        }
    };
}

implement_macro_for_all_value_types_and_2_typed_graphblas_functions_with_mutable_scalar_type_conversion!(
    implement_monoid_reducer,
    GrB_Matrix_reduce,
    GrB_Vector_reduce
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::operators::monoid::Plus as MonoidPlus;

    use crate::value_types::sparse_matrix::{FromMatrixElementList, MatrixElementList, Size};
    use crate::value_types::sparse_vector::{
        FromVectorElementList, GetVectorElementValue, VectorElementList,
    };
    use crate::value_types::utilities_to_implement_traits_for_all_value_types::implement_macro_for_all_value_types_except_bool;

    macro_rules! test_monoid {
        ($value_type:ty) => {
            paste::paste! {
                #[test]
                fn [<test_monoid_to_vector_reducer_ $value_type>]() {
                    let context = Context::init_ready(Mode::NonBlocking).unwrap();

                    let element_list = MatrixElementList::<$value_type>::from_element_vector(vec![
                        (1, 1, 1 as $value_type).into(),
                        (1, 5, 1 as $value_type).into(),
                        (2, 1, 2 as $value_type).into(),
                        (4, 2, 4 as $value_type).into(),
                        (5, 2, 5 as $value_type).into(),
                    ]);

                    let matrix_size: Size = (10, 15).into();
                    let matrix = SparseMatrix::<$value_type>::from_element_list(
                        &context.clone(),
                        &matrix_size,
                        &element_list,
                        &First::<$value_type, $value_type, $value_type>::new(),
                    )
                    .unwrap();

                    let mut product_vector =
                        SparseVector::<$value_type>::new(&context, &matrix_size.row_height()).unwrap();

                    let reducer = MonoidReducer::new(
                        &MonoidPlus::<$value_type>::new(),
                        &OperatorOptions::new_default(),
                        None,
                    );

                    reducer.to_vector(&matrix, &mut product_vector).unwrap();

                    println!("{}", product_vector);

                    assert_eq!(product_vector.number_of_stored_elements().unwrap(), 4);
                    assert_eq!(product_vector.get_element_value(&1).unwrap(), 2 as $value_type);
                    assert_eq!(product_vector.get_element_value(&2).unwrap(), 2 as $value_type);
                    assert_eq!(product_vector.get_element_value(&9).unwrap(), 0 as $value_type);

                    let mask_element_list = VectorElementList::<$value_type>::from_element_vector(vec![
                        (1, 1 as $value_type).into(),
                        (2, 2 as $value_type).into(),
                        (4, 4 as $value_type).into(),
                        // (5, 5).into(),
                    ]);

                    let mask = SparseVector::<$value_type>::from_element_list(
                        &context.clone(),
                        &matrix_size.row_height(),
                        &mask_element_list,
                        &First::<$value_type, $value_type, $value_type>::new(),
                    )
                    .unwrap();

                    let mut product_vector =
                        SparseVector::<$value_type>::new(&context, &matrix_size.row_height()).unwrap();

                    reducer
                        .to_vector_with_mask(&matrix, &mut product_vector, &mask)
                        .unwrap();

                    println!("{}", matrix);
                    println!("{}", product_vector);

                    assert_eq!(product_vector.number_of_stored_elements().unwrap(), 3);
                    assert_eq!(product_vector.get_element_value(&1).unwrap(), 2 as $value_type);
                    assert_eq!(product_vector.get_element_value(&2).unwrap(), 2 as $value_type);
                    assert_eq!(product_vector.get_element_value(&5).unwrap(), 0 as $value_type);
                    assert_eq!(product_vector.get_element_value(&9).unwrap(), 0 as $value_type);
                }

                #[test]
                fn [<test_monoid_to_scalar_reducer_for_matrix_ $value_type>]() {
                    let context = Context::init_ready(Mode::NonBlocking).unwrap();

                    let element_list = MatrixElementList::<$value_type>::from_element_vector(vec![
                        (1, 1, 1 as $value_type).into(),
                        (1, 5, 1 as $value_type).into(),
                        (2, 1, 2 as $value_type).into(),
                        (4, 2, 4 as $value_type).into(),
                        (5, 2, 5 as $value_type).into(),
                    ]);

                    let matrix_size: Size = (10, 15).into();
                    let matrix = SparseMatrix::<$value_type>::from_element_list(
                        &context.clone(),
                        &matrix_size,
                        &element_list,
                        &First::<$value_type, $value_type, $value_type>::new(),
                    )
                    .unwrap();

                    let mut product = 1 as $value_type;

                    let reducer = MonoidReducer::new(
                        &MonoidPlus::<$value_type>::new(),
                        &OperatorOptions::new_default(),
                        None,
                    );

                    reducer.matrix_to_scalar(&matrix, &mut product).unwrap();

                    println!("{}", product);

                    assert_eq!(product, 13 as $value_type);
                }

                #[test]
                fn [<test_monoid_to_scalar_reducer_for_vector_ $value_type>]() {
                    let context = Context::init_ready(Mode::NonBlocking).unwrap();
            
                    let element_list = VectorElementList::<$value_type>::from_element_vector(vec![
                        (1, 1 as $value_type).into(),
                        (2, 2 as $value_type).into(),
                        (4, 4 as $value_type).into(),
                        (5, 5 as $value_type).into(),
                    ]);
            
                    let vector_length = 10;
                    let vector = SparseVector::<$value_type>::from_element_list(
                        &context.clone(),
                        &vector_length,
                        &element_list,
                        &First::<$value_type, $value_type, $value_type>::new(),
                    )
                    .unwrap();
            
                    let mut product = 0 as $value_type;
            
                    let reducer = MonoidReducer::new(
                        &MonoidPlus::<$value_type>::new(),
                        &OperatorOptions::new_default(),
                        None,
                    );
            
                    reducer.vector_to_scalar(&vector, &mut product).unwrap();
            
                    println!("{}", product);
            
                    assert_eq!(product, 12 as $value_type);
                }
            }
        };
    }

    implement_macro_for_all_value_types_except_bool!(test_monoid);
}
