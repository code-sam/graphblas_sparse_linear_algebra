mod element_wise_matrix_multiplication;
mod element_wise_vector_multiplication;

pub use element_wise_matrix_multiplication::{
    ElementWiseMatrixMultiplicationBinaryOperator, ElementWiseMatrixMultiplicationMonoidOperator,
    ElementWiseMatrixMultiplicationSemiring,
};
pub use element_wise_vector_multiplication::{
    ElementWiseVectorMultiplicationBinaryOperator, ElementWiseVectorMultiplicationMonoidOperator,
    ElementWiseVectorMultiplicationSemiring,
};
