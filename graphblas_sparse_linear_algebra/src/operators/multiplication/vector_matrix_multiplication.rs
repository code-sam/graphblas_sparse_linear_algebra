use std::marker::PhantomData;
use std::ptr;

use crate::collections::sparse_matrix::{GraphblasSparseMatrixTrait, SparseMatrix};
use crate::collections::sparse_vector::{
    GraphblasSparseVectorTrait, SparseVector, SparseVectorTrait,
};
use crate::context::{CallGraphBlasContext, ContextTrait};
use crate::error::SparseLinearAlgebraError;
use crate::operators::binary_operator::BinaryOperator;
use crate::operators::options::OperatorOptions;
use crate::operators::semiring::Semiring;
use crate::value_type::utilities_to_implement_traits_for_all_value_types::{
    implement_trait_for_3_type_data_type_and_all_value_types,
    implement_trait_for_4_type_data_type_and_all_value_types,
};
use crate::value_type::{AsBoolean, ValueType};

use crate::bindings_to_graphblas_implementation::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Semiring, GrB_vxm,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl<
        FirstArgument: ValueType,
        SecondArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > Send
    for VectorMatrixMultiplicationOperator<FirstArgument, SecondArgument, Product, EvaluationDomain>
{
}
unsafe impl<
        FirstArgument: ValueType,
        SecondArgument: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > Sync
    for VectorMatrixMultiplicationOperator<FirstArgument, SecondArgument, Product, EvaluationDomain>
{
}

// TODO: review the use of &'a dyn Trait, removing dynamic dispatch could provide a performance gain. (it might be negated if cloning is necessary though)
// https://www.joshmcguigan.com/blog/cost-of-indirection-rust/
#[derive(Debug, Clone)]
pub struct VectorMatrixMultiplicationOperator<Multiplier, Multiplicant, Product, EvaluationDomain>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
{
    _multiplier: PhantomData<Multiplier>,
    _multiplicant: PhantomData<Multiplicant>,
    _product: PhantomData<Product>,
    _evaluation_domain: PhantomData<EvaluationDomain>,

    accumulator: GrB_BinaryOp, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
    semiring: GrB_Semiring,    // defines '+' and '*' for A*B (not optional for GrB_mxm)
    options: GrB_Descriptor,
}

impl<Multiplier, Multiplicant, Product, EvaluationDomain>
    VectorMatrixMultiplicationOperator<Multiplier, Multiplicant, Product, EvaluationDomain>
where
    Multiplier: ValueType,
    Multiplicant: ValueType,
    Product: ValueType,
    EvaluationDomain: ValueType,
{
    pub fn new(
        semiring: Box<dyn Semiring<Multiplier, Multiplicant, Product, EvaluationDomain>>, // defines '+' and '*' for A*B (not optional for GrB_mxm)
        options: OperatorOptions,
        accumulator: Option<Box<dyn BinaryOperator<Product, Product, Product, Product>>>, // optional accum for Z=accum(C,T), determines how results are written into the result matrix C
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
            _evaluation_domain: PhantomData,
        }
    }
}

pub trait MultiplyVectorByMatrix<Multiplier: ValueType, Multiplicant: ValueType, Product: ValueType>
{
    // TODO: consider a version where the resulting product matrix is generated in the function body
    fn apply(
        &self,
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseMatrix<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;

    fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        mask: &SparseVector<MaskValueType>,
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseMatrix<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl<
        Multiplier: ValueType,
        Multiplicant: ValueType,
        Product: ValueType,
        EvaluationDomain: ValueType,
    > MultiplyVectorByMatrix<Multiplier, Multiplicant, Product>
    for VectorMatrixMultiplicationOperator<Multiplier, Multiplicant, Product, EvaluationDomain>
{
    // TODO: consider a version where the resulting product matrix is generated in the function body
    fn apply(
        &self,
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseMatrix<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_vxm(
                    product.graphblas_vector(),
                    ptr::null_mut(),
                    self.accumulator,
                    self.semiring,
                    multiplier.graphblas_vector(),
                    multiplicant.graphblas_matrix(),
                    self.options,
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
    }

    fn apply_with_mask<MaskValueType: ValueType + AsBoolean>(
        &self,
        mask: &SparseVector<MaskValueType>,
        multiplier: &SparseVector<Multiplier>,
        multiplicant: &SparseMatrix<Multiplicant>,
        product: &mut SparseVector<Product>,
    ) -> Result<(), SparseLinearAlgebraError> {
        let context = product.context();

        context.call(
            || unsafe {
                GrB_vxm(
                    product.graphblas_vector(),
                    mask.graphblas_vector(),
                    self.accumulator,
                    self.semiring,
                    multiplier.graphblas_vector(),
                    multiplicant.graphblas_matrix(),
                    self.options,
                )
            },
            unsafe { product.graphblas_vector_ref() },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collections::sparse_matrix::{FromMatrixElementList, MatrixElementList, Size};
    use crate::collections::sparse_vector::{
        FromVectorElementList, GetVectorElementList, GetVectorElementValue, VectorElementList,
    };
    use crate::collections::Collection;
    use crate::context::{Context, Mode};
    use crate::operators::binary_operator::First;
    use crate::operators::binary_operator::Plus;
    use crate::operators::semiring::PlusTimes;

    #[test]
    fn test_multiplication_with_plus_times() {
        let context = Context::init_ready(Mode::NonBlocking).unwrap();

        let semiring = Box::new(PlusTimes::<f32, f32, f32, f32>::new());
        let options = OperatorOptions::new_default();
        let matrix_multiplier = VectorMatrixMultiplicationOperator::<f32, f32, f32, f32>::new(
            semiring.clone(),
            options.clone(),
            None,
        );

        let length = 2;
        let size: Size = (length, length).into();

        let multiplier = SparseVector::<f32>::new(&context, &length).unwrap();
        let multiplicant = SparseMatrix::<f32>::new(&context, &size).unwrap();
        let mut product = multiplier.clone();

        // Test multiplication of empty matrices
        matrix_multiplier
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();
        let element_list = product.get_element_list().unwrap();

        assert_eq!(product.number_of_stored_elements().unwrap(), 0);
        assert_eq!(element_list.length(), 0);
        assert_eq!(product.get_element_value(&1).unwrap(), 0.); // NoValue

        let multiplier_element_list =
            VectorElementList::<f32>::from_element_vector(vec![(0, 1.0).into(), (1, 2.0).into()]);
        let multiplier = SparseVector::<f32>::from_element_list(
            &context,
            &length,
            &multiplier_element_list,
            &First::<f32, f32, f32, f32>::new(),
        )
        .unwrap();

        let multiplicant_element_list = MatrixElementList::<f32>::from_element_vector(vec![
            (0, 0, 5.0).into(),
            (1, 0, 6.0).into(),
            (0, 1, 7.0).into(),
            (1, 1, 8.0).into(),
        ]);
        let multiplicant = SparseMatrix::<f32>::from_element_list(
            &context,
            &size,
            &multiplicant_element_list,
            &First::<f32, f32, f32, f32>::new(),
        )
        .unwrap();

        // Test multiplication of full matrices
        matrix_multiplier
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 17.);
        assert_eq!(product.get_element_value(&1).unwrap(), 23.);

        // TODO: this test is not generic over column/row storage format.
        // Equality checks should be done at a matrix level, since the ordering of the element list is not guaranteed.
        let expected_product =
            VectorElementList::<f32>::from_element_vector(vec![(0, 17.).into(), (1, 23.).into()]);
        let product_element_list = product.get_element_list().unwrap();
        assert_eq!(expected_product, product_element_list);

        // test the use of an accumulator
        let accumulator = Box::new(Plus::<f32, f32, f32, f32>::new());
        let matrix_multiplier_with_accumulator =
            VectorMatrixMultiplicationOperator::<f32, f32, f32, f32>::new(
                semiring.clone(),
                options.clone(),
                Some(accumulator),
            );

        matrix_multiplier_with_accumulator
            .apply(&multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 17. * 2.);
        assert_eq!(product.get_element_value(&1).unwrap(), 23. * 2.);

        // test the use of a mask
        let mask_element_list =
            VectorElementList::<u8>::from_element_vector(vec![(0, 3).into(), (1, 0).into()]);
        let mask = SparseVector::<u8>::from_element_list(
            &context,
            &length,
            &mask_element_list,
            &First::<u8, u8, u8, u8>::new(),
        )
        .unwrap();

        let matrix_multiplier = VectorMatrixMultiplicationOperator::<f32, f32, f32, f32>::new(
            semiring.clone(),
            options.clone(),
            None,
        );

        let mut product = SparseVector::<f32>::new(&context, &length).unwrap();

        matrix_multiplier
            .apply_with_mask(&mask, &multiplier, &multiplicant, &mut product)
            .unwrap();

        assert_eq!(product.get_element_value(&0).unwrap(), 17.);
        assert_eq!(product.get_element_value(&1).unwrap(), 0.);
    }
}
