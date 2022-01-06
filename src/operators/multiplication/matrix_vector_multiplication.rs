use std::ptr;

use std::marker::PhantomData;

use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::BinaryOperator;
use crate::operators::options::OperatorOptions;
use crate::operators::semiring::Semiring;
use crate::value_types::sparse_matrix::SparseMatrix;
use crate::value_types::sparse_vector::SparseVector;
use crate::value_types::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Semiring, GrB_mxv,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for MatrixVectorMultiplicationOperator<bool, bool, bool> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<u8, u8, u8> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<u16, u16, u16> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<u32, u32, u32> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<u64, u64, u64> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<i8, i8, i8> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<i16, i16, i16> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<i32, i32, i32> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<i64, i64, i64> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<f32, f32, f32> {}
unsafe impl Send for MatrixVectorMultiplicationOperator<f64, f64, f64> {}

unsafe impl Sync for MatrixVectorMultiplicationOperator<bool, bool, bool> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<u8, u8, u8> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<u16, u16, u16> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<u32, u32, u32> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<u64, u64, u64> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<i8, i8, i8> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<i16, i16, i16> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<i32, i32, i32> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<i64, i64, i64> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<f32, f32, f32> {}
unsafe impl Sync for MatrixVectorMultiplicationOperator<f64, f64, f64> {}

// TODO: review the use of &'a dyn Trait, removing dynamic dispatch could provide a performance gain. (it might be negated if cloning is necessary though)
// https://www.joshmcguigan.com/blog/cost-of-indirection-rust/
#[derive(Debug, Clone)]
pub struct MatrixVectorMultiplicationOperator<Multiplier, Multiplicant, Product>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
{
    _multiplier: PhantomData<Multiplier>,
    _multiplicant: PhantomData<Multiplicant>,
    _product: PhantomData<Product>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    semiring: GrB_Semiring,    // defines '+' and '*' for A*B (not optional for GrB_mxm)
    options: GrB_Descriptor,
}

impl<Multiplier, Multiplicant, Product>
    MatrixVectorMultiplicationOperator<Multiplier, Multiplicant, Product>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
{
    pub fn new(
        semiring: &dyn Semiring<Multiplier, Multiplicant, Product>, // defines '+' and '*' for A*B (not optional for GrB_mxm)
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
            semiring: semiring.graphblas_type(),
            options: options.to_graphblas_descriptor(),

            _multiplier: PhantomData,
            _multiplicant: PhantomData,
            _product: PhantomData,
        }
    }

    // TODO: consider a version where the resulting product matrix is generated in the function body
    pub fn apply(
        &self,
        // mask: Option<&SparseMatrix<AsBoolean<ValueType>>>,
        multiplier: &SparseMatrix<Multiplier>,
        multiplicant: &SparseVector<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_mxv(
                product.graphblas_vector(),
                ptr::null_mut(),
                self.accumulator,
                self.semiring,
                multiplier.graphblas_matrix(),
                multiplicant.graphblas_vector(),
                self.options,
            )
        })?;

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        mask: &SparseVector<AsBool>,
        multiplier: &SparseMatrix<Multiplier>,
        multiplicant: &SparseVector<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_mxv(
                product.graphblas_vector(),
                mask.graphblas_vector(),
                self.accumulator,
                self.semiring,
                multiplier.graphblas_matrix(),
                multiplicant.graphblas_vector(),
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
    use crate::operators::binary_operator::First;
    use crate::operators::binary_operator::Plus;
    use crate::operators::semiring::PlusTimes;
    use crate::value_types::sparse_matrix::{FromMatrixElementList, MatrixElementList, Size};
    use crate::value_types::sparse_vector::{
        FromVectorElementList, GetVectorElementList, GetVectorElementValue, VectorElementList,
    };

    #[test]
    fn test_multiplication_with_plus_times() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let semiring = PlusTimes::<f32, f32, f32>::new();
        let options = OperatorOptions::new_default();
        let matrix_multiplier =
            MatrixVectorMultiplicationOperator::<f32, f32, f32>::new(&semiring, &options, None);

        let length = 2;
        let size: Size = (length, length).into();

        let multiplier = SparseMatrix::<f32>::new(&context, &size).unwrap();
        let multiplicant = SparseVector::<f32>::new(&context, &length).unwrap();
        let mut product = multiplicant.clone();

        // Test multiplication of empty matrices
        matrix_multiplier
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();
        let element_list = product.get_element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.get_element_value(&1).unwrap(), 0.); // NoValue

        let multiplicant_element_list =
            VectorElementList::<f32>::from_element_vector(vec![(0, 1.0).into(), (1, 2.0).into()]);
        let multiplicant = SparseVector::<f32>::from_element_list(
            &context,
            &length,
            &multiplicant_element_list,
            &First::<f32, f32, f32>::new(),
        )
        .unwrap();

        let multiplier_element_list = MatrixElementList::<f32>::from_element_vector(vec![
            (0, 0, 5.0).into(),
            (1, 0, 6.0).into(),
            (0, 1, 7.0).into(),
            (1, 1, 8.0).into(),
        ]);
        let multiplier = SparseMatrix::<f32>::from_element_list(
            &context,
            &size,
            &multiplier_element_list,
            &First::<f32, f32, f32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        matrix_multiplier
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 19.);
        assert_eq!(product.get_element_value(&1).unwrap(), 22.);

        // TODO: this test is not generic over column/row storage format.
        // Equality checks should be done at a matrix level, since the ordering of the element list is not guaranteed.
        let expected_product =
            VectorElementList::<f32>::from_element_vector(vec![(0, 19.).into(), (1, 22.).into()]);
        let product_element_list = product.get_element_list().unwrap();
        assert_eq!(expected_product, product_element_list);

        // test the use of an accumulator
        let accumulator = Plus::<f32, f32, f32>::new();
        let matrix_multiplier_with_accumulator =
            MatrixVectorMultiplicationOperator::<f32, f32, f32>::new(
                &semiring,
                &options,
                Some(&accumulator),
            );

        matrix_multiplier_with_accumulator
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 19. * 2.);
        assert_eq!(product.get_element_value(&1).unwrap(), 22. * 2.);

        // test the use of a mask
        let mask_element_list =
            VectorElementList::<u8>::from_element_vector(vec![(0, 3).into(), (1, 0).into()]);
        let mask = SparseVector::<u8>::from_element_list(
            &context,
            &length,
            &mask_element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let matrix_multiplier =
            MatrixVectorMultiplicationOperator::<f32, f32, f32>::new(&semiring, &options, None);

        let mut product = SparseVector::<f32>::new(&context, &length).unwrap();

        matrix_multiplier
            .apply_with_mask(&mask, &multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 19.);
        assert_eq!(product.get_element_value(&1).unwrap(), 0.);
    }
}
