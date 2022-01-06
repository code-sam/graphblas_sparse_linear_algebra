use std::marker::PhantomData;
use std::ptr;

use crate::error::SparseLinearAlgebraError;
use crate::operators::{
    binary_operator::BinaryOperator, monoid::Monoid, options::OperatorOptions, semiring::Semiring,
};
use crate::value_types::sparse_vector::SparseVector;
use crate::value_types::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Monoid, GrB_Semiring, GrB_Vector_eWiseAdd_BinaryOp,
    GrB_Vector_eWiseAdd_Monoid, GrB_Vector_eWiseAdd_Semiring,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for ElementWiseVectorAdditionSemiring<bool, bool, bool> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<u8, u8, u8> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<u16, u16, u16> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<u32, u32, u32> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<u64, u64, u64> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<i8, i8, i8> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<i16, i16, i16> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<i32, i32, i32> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<i64, i64, i64> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<f32, f32, f32> {}
unsafe impl Send for ElementWiseVectorAdditionSemiring<f64, f64, f64> {}

unsafe impl Sync for ElementWiseVectorAdditionSemiring<bool, bool, bool> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<u8, u8, u8> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<u16, u16, u16> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<u32, u32, u32> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<u64, u64, u64> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<i8, i8, i8> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<i16, i16, i16> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<i32, i32, i32> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<i64, i64, i64> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<f32, f32, f32> {}
unsafe impl Sync for ElementWiseVectorAdditionSemiring<f64, f64, f64> {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorAdditionSemiring<Multiplier, Multiplicant, Product>
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
    ElementWiseVectorAdditionSemiring<Multiplier, Multiplicant, Product>
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
            GrB_Vector_eWiseAdd_Semiring(
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
        mask: &SparseVector<AsBool>,
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseVector<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseAdd_Semiring(
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
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<bool> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<u8> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<u16> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<u32> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<u64> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<i8> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<i16> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<i32> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<i64> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<f32> {}
unsafe impl Send for ElementWiseVectorAdditionMonoidOperator<f64> {}

unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<bool> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<u8> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<u16> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<u32> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<u64> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<i8> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<i16> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<i32> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<i64> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<f32> {}
unsafe impl Sync for ElementWiseVectorAdditionMonoidOperator<f64> {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorAdditionMonoidOperator<T: ValueType> {
    _value: PhantomData<T>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    multiplication_operator: GrB_Monoid, // defines element-wise multiplication operator Multiplier.*Multiplicant
    options: GrB_Descriptor,
}

impl<T: ValueType> ElementWiseVectorAdditionMonoidOperator<T> {
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

    pub fn apply(
        &self,
        multiplier: &SparseVector<T>,
        multiplicant: &SparseVector<T>,
        product: &mut SparseVector<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseAdd_Monoid(
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
        mask: &SparseVector<AsBool>,
        multiplier: &SparseVector<T>,
        multiplicant: &SparseVector<T>,
        product: &mut SparseVector<T>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseAdd_Monoid(
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
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<bool, bool, bool> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<u8, u8, u8> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<u16, u16, u16> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<u32, u32, u32> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<u64, u64, u64> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<i8, i8, i8> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<i16, i16, i16> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<i32, i32, i32> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<i64, i64, i64> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<f32, f32, f32> {}
unsafe impl Send for ElementWiseVectorAdditionBinaryOperator<f64, f64, f64> {}

unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<bool, bool, bool> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<u8, u8, u8> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<u16, u16, u16> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<u32, u32, u32> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<u64, u64, u64> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<i8, i8, i8> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<i16, i16, i16> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<i32, i32, i32> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<i64, i64, i64> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<f32, f32, f32> {}
unsafe impl Sync for ElementWiseVectorAdditionBinaryOperator<f64, f64, f64> {}

#[derive(Debug, Clone)]
pub struct ElementWiseVectorAdditionBinaryOperator<Multiplier, Multiplicant, Product> {
    _multiplier: PhantomData<Multiplier>,
    _multiplicant: PhantomData<Multiplicant>,
    _product: PhantomData<Product>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    multiplication_operator: GrB_BinaryOp, // defines element-wise multiplication operator Multiplier.*Multiplicant
    options: GrB_Descriptor,
}

impl<Multiplier, Multiplicant, Product>
    ElementWiseVectorAdditionBinaryOperator<Multiplier, Multiplicant, Product>
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
            GrB_Vector_eWiseAdd_BinaryOp(
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
        mask: &SparseVector<AsBool>,
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseVector<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(|| unsafe {
            GrB_Vector_eWiseAdd_BinaryOp(
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
            ElementWiseVectorAdditionBinaryOperator::<i64, i64, i64>::new(
                &operator, &options, None,
            );

        let accumulator = Times::<i64, i64, i64>::new();

        let _matrix_multiplier = ElementWiseVectorAdditionBinaryOperator::<i64, i64, i64>::new(
            &operator,
            &options,
            Some(&accumulator),
        );
    }

    #[test]
    fn test_element_wise_multiplication() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let operator = Times::<i32, i32, i32>::new();
        let options = OperatorOptions::new_default();
        let element_wise_vector_multiplier =
            ElementWiseVectorAdditionBinaryOperator::<i32, i32, i32>::new(
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
            ElementWiseVectorAdditionBinaryOperator::<i32, i32, i32>::new(
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

        let matrix_multiplier = ElementWiseVectorAdditionBinaryOperator::<i32, i32, i32>::new(
            &operator, &options, None,
        );

        let mut product = SparseVector::<i32>::new(&context, &length).unwrap();

        matrix_multiplier
            .apply_with_mask(&mask, &multiplier, &multiplicant, &mut product)
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
