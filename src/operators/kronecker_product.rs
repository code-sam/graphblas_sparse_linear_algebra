use std::ptr;

use std::marker::PhantomData;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator, mask::MatrixMask, monoid::Monoid, options::OperatorOptions,
    semiring::Semiring,
};
use crate::sparse_matrix::SparseMatrix;
use crate::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Matrix_kronecker_BinaryOp, GrB_Matrix_kronecker_Monoid,
    GrB_Matrix_kronecker_Semiring, GrB_Monoid, GrB_Semiring,
};

#[derive(Debug, Clone)]
pub struct SemiringKroneckerProduct<Multiplier, Multiplicant, Product>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
{
    _multiplier: PhantomData<Multiplier>,
    _multiplicant: PhantomData<Multiplicant>,
    _product: PhantomData<Product>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    multiplication_operator: GrB_Semiring, // defines element-wise multiplication operator Multiplier.*Multiplicant
    options: GrB_Descriptor,
}

impl<Multiplier, Multiplicant, Product> SemiringKroneckerProduct<Multiplier, Multiplicant, Product>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
{
    pub fn new(
        multiplication_operator: &dyn Semiring<Multiplier, Multiplicant, Product>, // defines element-wise multiplication operator Multiplier.*Multiplicant
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<Product, Product, Product>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            accumulator: accumulator_to_use,
            multiplication_operator: multiplication_operator.graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _multiplier: PhantomData,
            _multiplicant: PhantomData,
            _product: PhantomData,
        }
    }

    pub fn apply(
        &self,
        multiplier: &SparseMatrix<Multiplier>,
        multiplicant: &SparseMatrix<Multiplicant>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_kronecker_Semiring(
                product.graphblas_matrix(),
                ptr::null_mut(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_matrix(),
                multiplicant.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        mask: &MatrixMask<MaskValueType, AsBool>,
        multiplier: &SparseMatrix<Multiplier>,
        multiplicant: &SparseMatrix<Multiplicant>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_kronecker_Semiring(
                product.graphblas_matrix(),
                mask.graphblas_matrix(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_matrix(),
                multiplicant.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MonoidKroneckerProduct<T: ValueType> {
    _value: PhantomData<T>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    multiplication_operator: GrB_Monoid, // defines element-wise multiplication operator Multiplier.*Multiplicant
    options: GrB_Descriptor,
}

impl<T: ValueType> MonoidKroneckerProduct<T> {
    pub fn new(
        multiplication_operator: &dyn Monoid<T>, // defines element-wise multiplication operator Multiplier.*Multiplicant
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
            multiplication_operator: multiplication_operator.graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _value: PhantomData,
        }
    }

    pub fn apply<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        multiplier: &SparseMatrix<T>,
        multiplicant: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_kronecker_Monoid(
                product.graphblas_matrix(),
                ptr::null_mut(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_matrix(),
                multiplicant.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }

    fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        mask: &MatrixMask<MaskValueType, AsBool>,
        multiplier: &SparseMatrix<T>,
        multiplicant: &SparseMatrix<T>,
        product: &mut SparseMatrix<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_kronecker_Monoid(
                product.graphblas_matrix(),
                mask.graphblas_matrix(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_matrix(),
                multiplicant.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BinaryOperatorKroneckerProductOperator<Multiplier, Multiplicant, Product> {
    _multiplier: PhantomData<Multiplier>,
    _multiplicant: PhantomData<Multiplicant>,
    _product: PhantomData<Product>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    multiplication_operator: GrB_BinaryOp, // defines element-wise multiplication operator Multiplier.*Multiplicant
    options: GrB_Descriptor,
}

impl<Multiplier, Multiplicant, Product>
    BinaryOperatorKroneckerProductOperator<Multiplier, Multiplicant, Product>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
{
    pub fn new(
        multiplication_operator: &dyn BinaryOperator<Multiplier, Multiplicant, Product>, // defines element-wise multiplication operator Multiplier.*Multiplicant
        options: &OperatorOptions,
        accumulator: Option<&dyn BinaryOperator<Product, Product, Product>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    ) -> Self {
        let accumulator_to_use;
        match accumulator {
            Some(accumulator) => accumulator_to_use = accumulator.graphblas_type(),
            None => accumulator_to_use = ptr::null_mut(),
        }

        Self {
            accumulator: accumulator_to_use,
            multiplication_operator: multiplication_operator.graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _multiplier: PhantomData,
            _multiplicant: PhantomData,
            _product: PhantomData,
        }
    }

    pub fn apply(
        &self,
        multiplier: &SparseMatrix<Multiplier>,
        multiplicant: &SparseMatrix<Multiplicant>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_kronecker_BinaryOp(
                product.graphblas_matrix(),
                ptr::null_mut(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_matrix(),
                multiplicant.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        mask: &MatrixMask<MaskValueType, AsBool>,
        multiplier: &SparseMatrix<Multiplier>,
        multiplicant: &SparseMatrix<Multiplicant>,
        product: &mut SparseMatrix<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Matrix_kronecker_BinaryOp(
                product.graphblas_matrix(),
                mask.graphblas_matrix(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_matrix(),
                multiplicant.graphblas_matrix(),
                self.options,
            )
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::{First, Times};

    use crate::sparse_matrix::{
        FromMatrixElementList, GetMatrixElementList, GetMatrixElementValue, MatrixElementList, Size,
    };

    #[test]
    fn create_matrix_multiplier() {
        let operator = Times::<i64, i64, i64>::new();
        let options = OperatorOptions::new_default();
        let _element_wise_matrix_multiplier =
            BinaryOperatorKroneckerProductOperator::<i64, i64, i64>::new(&operator, &options, None);

        let _context = Context::init_ready(Mode::NonBlocking).unwrap();

        let target_height = 10;
        let target_width = 5;
        let _size: Size = (target_height, target_width).into();

        let accumulator = Times::<i64, i64, i64>::new();

        let _matrix_multiplier = BinaryOperatorKroneckerProductOperator::<i64, i64, i64>::new(
            &operator,
            &options,
            Some(&accumulator),
        );
    }

    #[test]
    fn test_element_wisemultiplication() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let operator = Times::<i32, i32, i32>::new();
        let options = OperatorOptions::new_default();
        let element_wise_matrix_multiplier =
            BinaryOperatorKroneckerProductOperator::<i32, i32, i32>::new(&operator, &options, None);

        let height = 2;
        let width = 2;
        let size: Size = (height, width).into();

        let multiplier = SparseMatrix::<i32>::new(&context, &size).unwrap();
        let multiplicant = multiplier.clone();

        let mut product = SparseMatrix::<i32>::new(&context, &(4, 4).into()).unwrap();

        // Test multiplication of empty matrices
        element_wise_matrix_multiplier
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();
        let element_list = product.get_element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.get_element_value(&(1, 1).into()).unwrap(), 0); // NoValue

        let multiplier_element_list = MatrixElementList::<i32>::from_element_vector(vec![
            (0, 0, 1).into(),
            (1, 0, 2).into(),
            (0, 1, 3).into(),
            (1, 1, 4).into(),
        ]);
        let multiplier = SparseMatrix::<i32>::from_element_list(
            &context,
            &size,
            &multiplier_element_list,
            &First::<i32, i32, i32>::new(),
        )
        .unwrap();

        let multiplicant_element_list = MatrixElementList::<i32>::from_element_vector(vec![
            (0, 0, 5).into(),
            (1, 0, 6).into(),
            (0, 1, 7).into(),
            (1, 1, 8).into(),
        ]);
        let multiplicant = SparseMatrix::<i32>::from_element_list(
            &context,
            &size,
            &multiplicant_element_list,
            &First::<i32, i32, i32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        element_wise_matrix_multiplier
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&(0, 0).into()).unwrap(), 5);
        assert_eq!(product.get_element_value(&(1, 0).into()).unwrap(), 6);
        assert_eq!(product.get_element_value(&(0, 1).into()).unwrap(), 7);
        assert_eq!(product.get_element_value(&(1, 1).into()).unwrap(), 8);

        assert_eq!(product.get_element_value(&(2, 0).into()).unwrap(), 10);
        assert_eq!(product.get_element_value(&(3, 0).into()).unwrap(), 12);
        assert_eq!(product.get_element_value(&(2, 1).into()).unwrap(), 14);
        assert_eq!(product.get_element_value(&(3, 1).into()).unwrap(), 16);

        assert_eq!(product.get_element_value(&(0, 2).into()).unwrap(), 15);
        assert_eq!(product.get_element_value(&(1, 2).into()).unwrap(), 18);
        assert_eq!(product.get_element_value(&(0, 3).into()).unwrap(), 21);
        assert_eq!(product.get_element_value(&(1, 3).into()).unwrap(), 24);

        assert_eq!(product.get_element_value(&(2, 2).into()).unwrap(), 20);
        assert_eq!(product.get_element_value(&(3, 2).into()).unwrap(), 24);
        assert_eq!(product.get_element_value(&(2, 3).into()).unwrap(), 28);
        assert_eq!(product.get_element_value(&(3, 3).into()).unwrap(), 32);

        // // test the use of an accumulator
        // let accumulator = Plus::<i32, i32, i32>::new();
        // let matrix_multiplier_with_accumulator =
        //     ElementWiseMatrixMultiplicationBinaryOperator::<i32, i32, i32>::new(
        //         &operator,
        //         &options,
        //         Some(&accumulator),
        //     );

        // matrix_multiplier_with_accumulator
        //     .apply(&multiplier, &multiplicant, &mut product)
        //     .unwrap();

        // assert_eq!(product.get_element_value((0, 0).into()).unwrap(), 5 * 2);
        // assert_eq!(product.get_element_value((1, 0).into()).unwrap(), 12 * 2);
        // assert_eq!(product.get_element_value((0, 1).into()).unwrap(), 21 * 2);
        // assert_eq!(product.get_element_value((1, 1).into()).unwrap(), 32 * 2);

        // // test the use of a mask
        // let mask_element_list = MatrixElementList::<u8>::from_element_vector(vec![
        //     (0, 0, 3).into(),
        //     (1, 0, 0).into(),
        //     (1, 1, 1).into(),
        // ]);
        // let mask = SparseMatrix::<u8>::from_element_list(
        //     &context,
        //     &size,
        //     &mask_element_list,
        //     &First::<u8, u8, u8>::new(),
        // )
        // .unwrap();

        // let matrix_multiplier = ElementWiseMatrixMultiplicationBinaryOperator::<i32, i32, i32>::new(
        //     &operator, &options, None,
        // );

        // let mut product = SparseMatrix::<i32>::new(&context, &size).unwrap();

        // matrix_multiplier
        //     .apply_with_mask(&mask.into(), &multiplier, &multiplicant, &mut product)
        //     .unwrap();

        // assert_eq!(product.get_element_value((0, 0).into()).unwrap(), 5);
        // assert_eq!(product.get_element_value((1, 0).into()).unwrap(), 0);
        // assert_eq!(product.get_element_value((0, 1).into()).unwrap(), 0);
        // assert_eq!(product.get_element_value((1, 1).into()).unwrap(), 32);
    }
}
