use std::marker::PhantomData;
use std::ptr;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator, mask::VectorMask, monoid::Monoid, options::OperatorOptions,
    semiring::Semiring,
};
use crate::value_types::sparse_vector::SparseVector;
use crate::value_types::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Monoid, GrB_Semiring, GrB_Vector_eWiseMult_BinaryOp,
    GrB_Vector_eWiseMult_Monoid, GrB_Vector_eWiseMult_Semiring,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<bool, bool, bool> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<u8, u8, u8> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<u16, u16, u16> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<u32, u32, u32> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<u64, u64, u64> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<i8, i8, i8> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<i16, i16, i16> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<i32, i32, i32> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<i64, i64, i64> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<f32, f32, f32> {}
unsafe impl Send for ElementWiseVectorMultiplicationSemiring<f64, f64, f64> {}

unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<bool, bool, bool> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<u8, u8, u8> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<u16, u16, u16> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<u32, u32, u32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<u64, u64, u64> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<i8, i8, i8> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<i16, i16, i16> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<i32, i32, i32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<i64, i64, i64> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<f32, f32, f32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationSemiring<f64, f64, f64> {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorMultiplicationSemiring<Multiplier, Multiplicant, Product>
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

impl<Multiplier, Multiplicant, Product>
    ElementWiseVectorMultiplicationSemiring<Multiplier, Multiplicant, Product>
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
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseVector<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseMult_Semiring(
                product.graphblas_vector(),
                ptr::null_mut(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_vector(),
                multiplicant.graphblas_vector(),
                self.options,
            )
        })?;

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        mask: &VectorMask<MaskValueType, AsBool>,
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseVector<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseMult_Semiring(
                product.graphblas_vector(),
                mask.graphblas_vector(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_vector(),
                multiplicant.graphblas_vector(),
                self.options,
            )
        })?;

        Ok(())
    }
}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<bool> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<u8> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<u16> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<u32> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<u64> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<i8> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<i16> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<i32> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<i64> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<f32> {}
unsafe impl Send for ElementWiseVectorMultiplicationMonoidOperator<f64> {}

unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<bool> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<u8> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<u16> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<u32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<u64> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<i8> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<i16> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<i32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<i64> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<f32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationMonoidOperator<f64> {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorMultiplicationMonoidOperator<T: ValueType> {
    _value: PhantomData<T>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    multiplication_operator: GrB_Monoid, // defines element-wise multiplication operator Multiplier.*Multiplicant
    options: GrB_Descriptor,
}

impl<T: ValueType> ElementWiseVectorMultiplicationMonoidOperator<T> {
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
        multiplier: &SparseVector<T>,
        multiplicant: &SparseVector<T>,
        product: &mut SparseVector<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseMult_Monoid(
                product.graphblas_vector(),
                ptr::null_mut(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_vector(),
                multiplicant.graphblas_vector(),
                self.options,
            )
        })?;

        Ok(())
    }

    fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        mask: &VectorMask<MaskValueType, AsBool>,
        multiplier: &SparseVector<T>,
        multiplicant: &SparseVector<T>,
        product: &mut SparseVector<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseMult_Monoid(
                product.graphblas_vector(),
                mask.graphblas_vector(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_vector(),
                multiplicant.graphblas_vector(),
                self.options,
            )
        })?;

        Ok(())
    }
}

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<bool, bool, bool> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<u8, u8, u8> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<u16, u16, u16> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<u32, u32, u32> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<u64, u64, u64> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<i8, i8, i8> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<i16, i16, i16> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<i32, i32, i32> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<i64, i64, i64> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<f32, f32, f32> {}
unsafe impl Send for ElementWiseVectorMultiplicationBinaryOperator<f64, f64, f64> {}

unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<bool, bool, bool> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<u8, u8, u8> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<u16, u16, u16> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<u32, u32, u32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<u64, u64, u64> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<i8, i8, i8> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<i16, i16, i16> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<i32, i32, i32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<i64, i64, i64> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<f32, f32, f32> {}
unsafe impl Sync for ElementWiseVectorMultiplicationBinaryOperator<f64, f64, f64> {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorMultiplicationBinaryOperator<Multiplier, Multiplicant, Product> {
    _multiplier: PhantomData<Multiplier>,
    _multiplicant: PhantomData<Multiplicant>,
    _product: PhantomData<Product>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    multiplication_operator: GrB_BinaryOp, // defines element-wise multiplication operator Multiplier.*Multiplicant
    options: GrB_Descriptor,
}

impl<Multiplier, Multiplicant, Product>
    ElementWiseVectorMultiplicationBinaryOperator<Multiplier, Multiplicant, Product>
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
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseVector<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseMult_BinaryOp(
                product.graphblas_vector(),
                ptr::null_mut(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_vector(),
                multiplicant.graphblas_vector(),
                self.options,
            )
        })?;

        Ok(())
    }

    pub fn apply_with_mask<MaskValueType: ValueType, AsBool: AsBoolean<MaskValueType>>(
        &self,
        mask: &VectorMask<MaskValueType, AsBool>,
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseVector<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseMult_BinaryOp(
                product.graphblas_vector(),
                mask.graphblas_vector(),
                self.accumulator,
                self.multiplication_operator,
                multiplier.graphblas_vector(),
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
    use crate::operators::binary_operator::{First, Plus, Times};
    use crate::value_types::sparse_vector::{
        FromVectorElementList, GetVectorElementList, GetVectorElementValue, VectorElementList,
    };

    #[test]
    fn create_vector_multiplier() {
        let operator = Times::<i64, i64, i64>::new();
        let options = OperatorOptions::new_default();
        let _element_wise_matrix_multiplier =
            ElementWiseVectorMultiplicationBinaryOperator::<i64, i64, i64>::new(
                &operator, &options, None,
            );

        let accumulator = Times::<i64, i64, i64>::new();

        let _matrix_multiplier =
            ElementWiseVectorMultiplicationBinaryOperator::<i64, i64, i64>::new(
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
        let element_wise_vector_multiplier =
            ElementWiseVectorMultiplicationBinaryOperator::<i32, i32, i32>::new(
                &operator, &options, None,
            );

        let length = 4;

        let multiplier = SparseVector::<i32>::new(&context, &length).unwrap();
        let multiplicant = multiplier.clone();
        let mut product = multiplier.clone();

        // Test multiplication of empty matrices
        element_wise_vector_multiplier
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();
        let element_list = product.get_element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.get_element_value(&1).unwrap(), 0); // NoValue

        let multiplier_element_list = VectorElementList::<i32>::from_element_vector(vec![
            (0, 1).into(),
            (1, 2).into(),
            (2, 3).into(),
            (3, 4).into(),
        ]);
        let multiplier = SparseVector::<i32>::from_element_list(
            &context,
            &length,
            &multiplier_element_list,
            &First::<i32, i32, i32>::new(),
        )
        .unwrap();

        let multiplicant_element_list = VectorElementList::<i32>::from_element_vector(vec![
            (0, 5).into(),
            (1, 6).into(),
            (2, 7).into(),
            (3, 8).into(),
        ]);
        let multiplicant = SparseVector::<i32>::from_element_list(
            &context,
            &length,
            &multiplicant_element_list,
            &First::<i32, i32, i32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        element_wise_vector_multiplier
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 5);
        assert_eq!(product.get_element_value(&1).unwrap(), 12);
        assert_eq!(product.get_element_value(&2).unwrap(), 21);
        assert_eq!(product.get_element_value(&3).unwrap(), 32);

        // test the use of an accumulator
        let accumulator = Plus::<i32, i32, i32>::new();
        let matrix_multiplier_with_accumulator =
            ElementWiseVectorMultiplicationBinaryOperator::<i32, i32, i32>::new(
                &operator,
                &options,
                Some(&accumulator),
            );

        matrix_multiplier_with_accumulator
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 5 * 2);
        assert_eq!(product.get_element_value(&1).unwrap(), 12 * 2);
        assert_eq!(product.get_element_value(&2).unwrap(), 21 * 2);
        assert_eq!(product.get_element_value(&3).unwrap(), 32 * 2);

        // test the use of a mask
        let mask_element_list = VectorElementList::<u8>::from_element_vector(vec![
            (0, 3).into(),
            (1, 0).into(),
            (3, 1).into(),
        ]);
        let mask = SparseVector::<u8>::from_element_list(
            &context,
            &length,
            &mask_element_list,
            &First::<u8, u8, u8>::new(),
        )
        .unwrap();

        let matrix_multiplier = ElementWiseVectorMultiplicationBinaryOperator::<i32, i32, i32>::new(
            &operator, &options, None,
        );

        let mut product = SparseVector::<i32>::new(&context, &length).unwrap();

        matrix_multiplier
            .apply_with_mask(&mask.into(), &multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 5);
        assert_eq!(product.get_element_value(&1).unwrap(), 0);
        assert_eq!(product.get_element_value(&2).unwrap(), 0);
        assert_eq!(product.get_element_value(&3).unwrap(), 32);
    }
}

// GB_PUBLIC
// GrB_Info GrB_Vector_eWiseMult_Semiring       // w<Mask> = accum (w, u.*v)
// (
//     GrB_Vector w,                   // input/output vector for results
//     const GrB_Vector mask,          // optional mask for w, unused if NULL
//     const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
//     const GrB_Semiring semiring,    // defines '.*' for t=u.*v
//     const GrB_Vector u,             // first input:  vector u
//     const GrB_Vector v,             // second input: vector v
//     const GrB_Descriptor desc       // descriptor for w and mask
// ) ;

// GB_PUBLIC
// GrB_Info GrB_Vector_eWiseMult_Monoid         // w<Mask> = accum (w, u.*v)
// (
//     GrB_Vector w,                   // input/output vector for results
//     const GrB_Vector mask,          // optional mask for w, unused if NULL
//     const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
//     const GrB_Monoid monoid,        // defines '.*' for t=u.*v
//     const GrB_Vector u,             // first input:  vector u
//     const GrB_Vector v,             // second input: vector v
//     const GrB_Descriptor desc       // descriptor for w and mask
// ) ;

// GB_PUBLIC
// GrB_Info GrB_Vector_eWiseMult_BinaryOp       // w<Mask> = accum (w, u.*v)
// (
//     GrB_Vector w,                   // input/output vector for results
//     const GrB_Vector mask,          // optional mask for w, unused if NULL
//     const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
//     const GrB_BinaryOp mult,        // defines '.*' for t=u.*v
//     const GrB_Vector u,             // first input:  vector u
//     const GrB_Vector v,             // second input: vector v
//     const GrB_Descriptor desc       // descriptor for w and mask
// ) ;
