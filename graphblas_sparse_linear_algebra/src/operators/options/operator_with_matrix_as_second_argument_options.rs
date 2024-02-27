use suitesparse_graphblas_sys::GrB_Descriptor;

use super::{
    graphblas_descriptor, GetClearOutputBeforeUse, GetGraphblasDescriptor,
    GetMaskedOperatorOptions, GetOperatorMaskOptions, GetOperatorOptions,
    GetTransposeMatrixArgument, GetTransposeSecondMatrixArgument, WithTransposeMatrixArgument,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for OperatorWithMatrixAsSecondArgumentOptions {}
unsafe impl Sync for OperatorWithMatrixAsSecondArgumentOptions {}

unsafe impl Send for MaskedOperatorWithMatrixAsSecondArgumentOptions {}
unsafe impl Sync for MaskedOperatorWithMatrixAsSecondArgumentOptions {}

#[derive(Debug, Clone)]
pub struct OperatorWithMatrixAsSecondArgumentOptions {
    clear_output_before_use: bool,
    transpose_matrix_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetOperatorWithMatrixAsSecondArgumentOptions:
    GetOperatorOptions + GetTransposeSecondMatrixArgument
{
}

impl GetOperatorOptions for OperatorWithMatrixAsSecondArgumentOptions {}
impl GetOperatorWithMatrixAsSecondArgumentOptions for OperatorWithMatrixAsSecondArgumentOptions {}

impl GetClearOutputBeforeUse for OperatorWithMatrixAsSecondArgumentOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetTransposeSecondMatrixArgument for OperatorWithMatrixAsSecondArgumentOptions {
    fn transpose_second_matrix_argument(&self) -> bool {
        self.transpose_matrix_argument
    }
}

impl GetGraphblasDescriptor for OperatorWithMatrixAsSecondArgumentOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeMatrixArgument for OperatorWithMatrixAsSecondArgumentOptions {
    fn with_negated_transpose_matrix_argument(&self) -> Self {
        OperatorWithMatrixAsSecondArgumentOptions::new(
            self.clear_output_before_use,
            !self.transpose_matrix_argument,
        )
    }

    fn with_transpose_matrix_argument(&self, transpose_matrix_argument: bool) -> Self {
        if transpose_matrix_argument == self.transpose_matrix_argument {
            self.to_owned()
        } else {
            OperatorWithMatrixAsSecondArgumentOptions::new(
                self.clear_output_before_use,
                transpose_matrix_argument,
            )
        }
    }
}

impl OperatorWithMatrixAsSecondArgumentOptions {
    pub fn new(clear_output_before_use: bool, transpose_matrix_argument: bool) -> Self {
        Self {
            clear_output_before_use,
            transpose_matrix_argument,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                false,
                transpose_matrix_argument,
            ),
        }
    }

    pub fn new_default() -> Self {
        let transpose_matrix_argument = false;
        let clear_output_before_use = false;

        Self {
            clear_output_before_use,
            transpose_matrix_argument,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                false,
                transpose_matrix_argument,
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaskedOperatorWithMatrixAsSecondArgumentOptions {
    clear_output_before_use: bool,
    use_mask_structure_of_stored_values_as_mask: bool,
    use_mask_complement: bool,
    transpose_matrix_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetMaskedOperatorWithMatrixAsSecondArgumentOptions:
    GetMaskedOperatorOptions + GetTransposeSecondMatrixArgument
{
}

impl GetMaskedOperatorOptions for MaskedOperatorWithMatrixAsSecondArgumentOptions {}
impl GetMaskedOperatorWithMatrixAsSecondArgumentOptions
    for MaskedOperatorWithMatrixAsSecondArgumentOptions
{
}

impl GetClearOutputBeforeUse for MaskedOperatorWithMatrixAsSecondArgumentOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetOperatorMaskOptions for MaskedOperatorWithMatrixAsSecondArgumentOptions {
    fn use_mask_structure_of_stored_values_as_mask(&self) -> bool {
        self.use_mask_structure_of_stored_values_as_mask
    }

    fn use_mask_complement(&self) -> bool {
        self.use_mask_complement
    }
}

impl GetTransposeSecondMatrixArgument for MaskedOperatorWithMatrixAsSecondArgumentOptions {
    fn transpose_second_matrix_argument(&self) -> bool {
        self.transpose_matrix_argument
    }
}

impl GetGraphblasDescriptor for MaskedOperatorWithMatrixAsSecondArgumentOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeMatrixArgument for MaskedOperatorWithMatrixAsSecondArgumentOptions {
    fn with_negated_transpose_matrix_argument(&self) -> Self {
        MaskedOperatorWithMatrixAsSecondArgumentOptions::new(
            self.clear_output_before_use,
            self.use_mask_structure_of_stored_values_as_mask,
            self.use_mask_complement,
            !self.transpose_matrix_argument,
        )
    }

    fn with_transpose_matrix_argument(&self, transpose_matrix: bool) -> Self {
        if transpose_matrix == self.transpose_matrix_argument {
            self.to_owned()
        } else {
            MaskedOperatorWithMatrixAsSecondArgumentOptions::new(
                self.clear_output_before_use,
                self.use_mask_structure_of_stored_values_as_mask,
                self.use_mask_complement,
                transpose_matrix,
            )
        }
    }
}

impl MaskedOperatorWithMatrixAsSecondArgumentOptions {
    pub fn new(
        clear_output_before_use: bool,
        use_mask_structure_of_stored_values_as_mask: bool,
        use_mask_complement: bool,
        transpose_matrix_argument: bool,
    ) -> Self {
        Self {
            clear_output_before_use,
            use_mask_structure_of_stored_values_as_mask,
            use_mask_complement,
            transpose_matrix_argument,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                false,
                transpose_matrix_argument,
            ),
        }
    }

    pub fn new_default() -> Self {
        let clear_output_before_use = false;
        let use_mask_structure_of_stored_values_as_mask = false;
        let use_mask_complement = false;
        let transpose_matrix_argument = false;

        Self {
            clear_output_before_use,
            use_mask_structure_of_stored_values_as_mask,
            use_mask_complement,
            transpose_matrix_argument,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                false,
                transpose_matrix_argument,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn test_options() {
        let default_options = MaskedOperatorWithMatrixAsSecondArgumentOptions::new_default();
        let expected_value: GrB_Descriptor = ptr::null_mut();
        assert_eq!(default_options.graphblas_descriptor(), expected_value)
    }
}
