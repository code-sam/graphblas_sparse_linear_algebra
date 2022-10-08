use std::ptr;

use std::marker::PhantomData;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{binary_operator::BinaryOperator, options::OperatorOptions};

use crate::value_types::sparse_matrix::SparseMatrix;
use crate::value_types::sparse_scalar::{SetScalarValue, SparseScalar};
use crate::value_types::utilities_to_implement_traits_for_all_value_types::{
    implement_macro_with_custom_input_version_1_for_all_value_types,
    implement_trait_for_all_value_types,
};
use crate::value_types::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GxB_DIAG, GxB_EQ_THUNK, GxB_EQ_ZERO, GxB_GE_THUNK, GxB_GE_ZERO,
    GxB_GT_THUNK, GxB_GT_ZERO, GxB_LE_THUNK, GxB_LE_ZERO, GxB_LT_THUNK, GxB_LT_ZERO,
    GxB_Matrix_select, GxB_NE_THUNK, GxB_NONZERO, GxB_OFFDIAG, GxB_TRIL, GxB_TRIU,
};

use super::diagonal_index::{DiagonalIndex, DiagonalIndexGraphblasType};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
implement_trait_for_all_value_types!(Send, MatrixSelector);
implement_trait_for_all_value_types!(Sync, MatrixSelector);

#[derive(Debug, Clone)]
pub struct MatrixSelector<T: ValueType> {
    _value: PhantomData<T>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    options: GrB_Descriptor,
}

impl<T: ValueType> MatrixSelector<T> {
    pub fn new(
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<T, T, T>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            accumulator: accumulator_to_use,
            options: options.to_graphblas_descriptor(),

            _value: PhantomData,
        }
    }
}

macro_rules! implement_selector_with_diagonal {
    ($method_name:ident, $method_name_with_mask:ident, $graphblas_operator:ident) => {
        impl<T: ValueType> MatrixSelector<T> {
            /// k = 0 selects the main diagonal, positive for above, negative for below
            pub fn $method_name(
                &self,
                argument: &SparseMatrix<T>,
                product: &mut SparseMatrix<T>,
                diagional: &DiagonalIndex,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let diagonal_index = diagional.to_graphblas_type(&context)?;

                match diagonal_index {
                    DiagonalIndexGraphblasType::Index(index) => {
                        context.call(|| unsafe {
                            GxB_Matrix_select(
                                product.graphblas_matrix(),
                                ptr::null_mut(),
                                self.accumulator,
                                $graphblas_operator,
                                argument.graphblas_matrix(),
                                index.graphblas_scalar(),
                                self.options,
                            )
                        })?;
                    }
                    DiagonalIndexGraphblasType::Default => {
                        context.call(|| unsafe {
                            GxB_Matrix_select(
                                product.graphblas_matrix(),
                                ptr::null_mut(),
                                self.accumulator,
                                $graphblas_operator,
                                argument.graphblas_matrix(),
                                ptr::null_mut(),
                                self.options,
                            )
                        })?;
                    }
                }

                Ok(())
            }

            /// k = 0 selects the main diagonal, positive for avove, negative for below
            pub fn $method_name_with_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                argument: &SparseMatrix<T>,
                product: &mut SparseMatrix<T>,
                diagional: &DiagonalIndex,
                mask: &SparseMatrix<AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let diagonal_index = diagional.to_graphblas_type(&context)?;

                match diagonal_index {
                    DiagonalIndexGraphblasType::Index(index) => {
                        context.call(|| unsafe {
                            GxB_Matrix_select(
                                product.graphblas_matrix(),
                                mask.graphblas_matrix(),
                                self.accumulator,
                                $graphblas_operator,
                                argument.graphblas_matrix(),
                                index.graphblas_scalar(),
                                self.options,
                            )
                        })?;
                    }
                    DiagonalIndexGraphblasType::Default => {
                        context.call(|| unsafe {
                            GxB_Matrix_select(
                                product.graphblas_matrix(),
                                mask.graphblas_matrix(),
                                self.accumulator,
                                $graphblas_operator,
                                argument.graphblas_matrix(),
                                ptr::null_mut(),
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

implement_selector_with_diagonal!(lower_triangle, lower_triangle_with_mask, GxB_TRIL);
implement_selector_with_diagonal!(upper_triangle, upper_triangle_with_mask, GxB_TRIU);
implement_selector_with_diagonal!(diagonal, diagonal_with_mask, GxB_DIAG);
implement_selector_with_diagonal!(clear_diagonal, clear_diagonal_with_mask, GxB_OFFDIAG);

macro_rules! implement_scalar_selector {
    ($value_type:ty, $selector_trait:ident, $method_name:ident, $method_name_with_mask:ident, $graphblas_operator:ident) => {
        impl $selector_trait<$value_type> for MatrixSelector<$value_type> {
            fn $method_name(
                &self,
                argument: &SparseMatrix<$value_type>,
                product: &mut SparseMatrix<$value_type>,
                scalar: &$value_type,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let mut sparse_scalar = SparseScalar::<$value_type>::new(&context)?;
                sparse_scalar.set_value(scalar)?;

                context.call(|| unsafe {
                    GxB_Matrix_select(
                        product.graphblas_matrix(),
                        ptr::null_mut(),
                        self.accumulator,
                        $graphblas_operator,
                        argument.graphblas_matrix(),
                        sparse_scalar.graphblas_scalar(),
                        self.options,
                    )
                })?;

                Ok(())
            }

            fn $method_name_with_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                argument: &SparseMatrix<$value_type>,
                product: &mut SparseMatrix<$value_type>,
                scalar: &$value_type,
                _mask: &SparseMatrix<AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();
                let mut sparse_scalar = SparseScalar::<$value_type>::new(&context)?;
                sparse_scalar.set_value(scalar)?;

                context.call(|| unsafe {
                    GxB_Matrix_select(
                        product.graphblas_matrix(),
                        ptr::null_mut(),
                        self.accumulator,
                        $graphblas_operator,
                        argument.graphblas_matrix(),
                        sparse_scalar.graphblas_scalar(),
                        self.options,
                    )
                })?;

                Ok(())
            }
        }
    };
}

pub trait SelectMatrixNotEqualToScalar<T: ValueType> {
    fn not_equal_to_scalar(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn not_equal_to_scalar_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
        mask: &SparseMatrix<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

implement_macro_with_custom_input_version_1_for_all_value_types!(
    implement_scalar_selector,
    SelectMatrixNotEqualToScalar,
    not_equal_to_scalar,
    not_equal_to_scalar_with_mask,
    GxB_NE_THUNK
);

pub trait SelectMatrixEqualToScalar<T: ValueType> {
    fn equal_to_scalar(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn equal_to_scalar_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
        mask: &SparseMatrix<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

implement_macro_with_custom_input_version_1_for_all_value_types!(
    implement_scalar_selector,
    SelectMatrixEqualToScalar,
    equal_to_scalar,
    equal_to_scalar_with_mask,
    GxB_EQ_THUNK
);

pub trait SelectMatrixGreaterThanScalar<T: ValueType> {
    fn greater_than_scalar(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn greater_than_scalar_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
        mask: &SparseMatrix<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

implement_macro_with_custom_input_version_1_for_all_value_types!(
    implement_scalar_selector,
    SelectMatrixGreaterThanScalar,
    greater_than_scalar,
    greater_than_scalar_with_mask,
    GxB_GT_THUNK
);

pub trait SelectMatrixGreaterThanOrEqualToScalar<T: ValueType> {
    fn greater_than_or_equal_to_scalar(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn greater_than_or_equal_to_scalar_with_mask<
        MaskValueType: ValueType,
        AsBool: AsBoolean<MaskValueType>,
    >(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
        mask: &SparseMatrix<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

implement_macro_with_custom_input_version_1_for_all_value_types!(
    implement_scalar_selector,
    SelectMatrixGreaterThanOrEqualToScalar,
    greater_than_or_equal_to_scalar,
    greater_than_or_equal_to_scalar_with_mask,
    GxB_GE_THUNK
);

pub trait SelectMatrixLessThanScalar<T: ValueType> {
    fn less_than_scalar(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn less_than_scalar_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
        mask: &SparseMatrix<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

implement_macro_with_custom_input_version_1_for_all_value_types!(
    implement_scalar_selector,
    SelectMatrixLessThanScalar,
    less_than_scalar,
    less_than_scalar_with_mask,
    GxB_LT_THUNK
);

pub trait SelectMatrixLessThanOrEqualToScalar<T: ValueType> {
    fn less_than_or_equal_to_scalar(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn less_than_less_than_or_equal_to_scalar_with_mask<
        MaskValueType: ValueType,
        AsBool: AsBoolean<MaskValueType>,
    >(
        &self,
        argument: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
        scalar: &T,
        mask: &SparseMatrix<AsBool>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

implement_macro_with_custom_input_version_1_for_all_value_types!(
    implement_scalar_selector,
    SelectMatrixLessThanOrEqualToScalar,
    less_than_or_equal_to_scalar,
    less_than_less_than_or_equal_to_scalar_with_mask,
    GxB_LE_THUNK
);

macro_rules! implement_selector_with_zero {
    ($method_name:ident, $method_name_with_mask:ident, $graphblas_operator:ident) => {
        impl<T: ValueType> MatrixSelector<T> {
            pub fn $method_name(
                &self,
                argument: &SparseMatrix<T>,
                product: &mut SparseMatrix<T>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                context.call(|| unsafe {
                    GxB_Matrix_select(
                        product.graphblas_matrix(),
                        ptr::null_mut(),
                        self.accumulator,
                        $graphblas_operator,
                        argument.graphblas_matrix(),
                        ptr::null_mut(),
                        self.options,
                    )
                })?;

                Ok(())
            }

            pub fn $method_name_with_mask<
                MaskValueType: ValueType,
                AsBool: AsBoolean<MaskValueType>,
            >(
                &self,
                argument: &SparseMatrix<T>,
                product: &mut SparseMatrix<T>,
                mask: &SparseMatrix<AsBool>,
            ) -> Result<(), SparseLinearAlgebraError> {
                let context = product.context();

                context.call(|| unsafe {
                    GxB_Matrix_select(
                        product.graphblas_matrix(),
                        mask.graphblas_matrix(),
                        self.accumulator,
                        $graphblas_operator,
                        argument.graphblas_matrix(),
                        ptr::null_mut(),
                        self.options,
                    )
                })?;

                Ok(())
            }
        }
    };
}

implement_selector_with_zero!(non_zero, non_zero_with_mask, GxB_NONZERO);
implement_selector_with_zero!(zero, zero_with_mask, GxB_EQ_ZERO);
implement_selector_with_zero!(positive, positive_with_mask, GxB_GT_ZERO);
implement_selector_with_zero!(zero_or_positive, zero_or_positive_with_mask, GxB_GE_ZERO);
implement_selector_with_zero!(negative, negative_with_mask, GxB_LT_ZERO);
implement_selector_with_zero!(zero_or_negative, zero_or_negative_with_mask, GxB_LE_ZERO);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;

    use crate::value_types::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };

    #[test]
    fn test_lower_triangle() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let selector = MatrixSelector::new(&OperatorOptions::new_default(), None);

        let diagonal_index = DiagonalIndex::Default();

        selector
            .lower_triangle(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);

        let diagonal_index = DiagonalIndex::Index(-1);
        // let diagonal_index = DiagonalIndex::Default();

        selector
            .lower_triangle(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 0);
    }

    #[test]
    fn test_upper_triangle() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let selector = MatrixSelector::new(&OperatorOptions::new_default(), None);

        let diagonal_index = DiagonalIndex::Default();

        selector
            .upper_triangle(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);

        let diagonal_index = DiagonalIndex::Index(-1);
        // let diagonal_index = DiagonalIndex::Default();

        selector
            .upper_triangle(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);
    }

    #[test]
    fn test_diagonal() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let selector = MatrixSelector::new(&OperatorOptions::new_default(), None);

        let diagonal_index = DiagonalIndex::Default();

        selector
            .diagonal(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);

        let diagonal_index = DiagonalIndex::Index(-1);
        // let diagonal_index = DiagonalIndex::Default();

        selector
            .diagonal(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 0);
    }

    #[test]
    fn test_clear_diagonal() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let selector = MatrixSelector::new(&OperatorOptions::new_default(), None);

        let diagonal_index = DiagonalIndex::Default();

        selector
            .clear_diagonal(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 0);

        let diagonal_index = DiagonalIndex::Index(-1);
        // let diagonal_index = DiagonalIndex::Default();

        selector
            .clear_diagonal(&matrix, &mut product_matrix, &diagonal_index)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);
    }

    #[test]
    fn test_zero_selector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let selector = MatrixSelector::new(&OperatorOptions::new_default(), None);

        selector.positive(&matrix, &mut product_matrix).unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 4);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);

        selector.negative(&matrix, &mut product_matrix).unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 0);
    }

    #[test]
    fn test_scalar_selector() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let element_list = MatrixElementList::<u8>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);

        let matrix_size: Size = (2, 2).into();
        let matrix = SparseMatrix::<u8>::from_element_list(
            &context.clone(),
            &matrix_size,
            &element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let mut product_matrix = SparseMatrix::<u8>::new(&context, &matrix_size).unwrap();

        let selector = MatrixSelector::new(&OperatorOptions::new_default(), None);

        selector
            .greater_than_scalar(&matrix, &mut product_matrix, &1)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 3);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 4);

        selector
            .less_than_scalar(&matrix, &mut product_matrix, &3)
            .unwrap();

        println!("{}", product_matrix);

        assert_eq!(product_matrix.number_of_stored_elements().unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 0).into()).unwrap(), 1);
        assert_eq!(product_matrix.get_element_value(&(1, 0).into()).unwrap(), 2);
        assert_eq!(product_matrix.get_element_value(&(0, 1).into()).unwrap(), 0);
        assert_eq!(product_matrix.get_element_value(&(1, 1).into()).unwrap(), 0);
    }
}
