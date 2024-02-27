use suitesparse_graphblas_sys::GrB_Descriptor;

use super::{
    graphblas_descriptor, GetClearOutputBeforeUse, GetGraphblasDescriptor,
    GetMaskedOperatorOptions, GetOperatorMaskOptions, GetOperatorOptions,
    GetTransposeMatrixArgument, WithTransposeMatrixArgument,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for OperatorWithMatrixArgumentOptions {}
unsafe impl Sync for OperatorWithMatrixArgumentOptions {}

unsafe impl Send for MaskedOperatorWithMatrixArgumentOptions {}
unsafe impl Sync for MaskedOperatorWithMatrixArgumentOptions {}

#[derive(Debug, Clone)]
pub struct OperatorWithMatrixArgumentOptions {
    clear_output_before_use: bool,
    transpose_matrix_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetOperatorWithMatrixArgumentOptions:
    GetOperatorOptions + GetTransposeMatrixArgument
{
}

impl GetOperatorOptions for OperatorWithMatrixArgumentOptions {}
impl GetOperatorWithMatrixArgumentOptions for OperatorWithMatrixArgumentOptions {}

impl GetClearOutputBeforeUse for OperatorWithMatrixArgumentOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetTransposeMatrixArgument for OperatorWithMatrixArgumentOptions {
    fn transpose_matrix_argument(&self) -> bool {
        self.transpose_matrix_argument
    }
}

impl GetGraphblasDescriptor for OperatorWithMatrixArgumentOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeMatrixArgument for OperatorWithMatrixArgumentOptions {
    fn with_negated_transpose_matrix_argument(&self) -> Self {
        OperatorWithMatrixArgumentOptions::new(
            self.clear_output_before_use,
            !self.transpose_matrix_argument,
        )
    }

    fn with_transpose_matrix_argument(&self, transpose_matrix_argument: bool) -> Self {
        if transpose_matrix_argument == self.transpose_matrix_argument {
            self.to_owned()
        } else {
            OperatorWithMatrixArgumentOptions::new(
                self.clear_output_before_use,
                transpose_matrix_argument,
            )
        }
    }
}

impl OperatorWithMatrixArgumentOptions {
    pub fn new(clear_output_before_use: bool, transpose_matrix_argument: bool) -> Self {
        Self {
            clear_output_before_use,
            transpose_matrix_argument,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                transpose_matrix_argument,
                false,
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
                transpose_matrix_argument,
                false,
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaskedOperatorWithMatrixArgumentOptions {
    clear_output_before_use: bool,
    use_mask_structure_of_stored_values_as_mask: bool,
    use_mask_complement: bool,
    transpose_matrix_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetMaskedOperatorWithMatrixArgumentOptions:
    GetMaskedOperatorOptions + GetTransposeMatrixArgument
{
}

impl GetMaskedOperatorOptions for MaskedOperatorWithMatrixArgumentOptions {}
impl GetMaskedOperatorWithMatrixArgumentOptions for MaskedOperatorWithMatrixArgumentOptions {}

impl GetClearOutputBeforeUse for MaskedOperatorWithMatrixArgumentOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetOperatorMaskOptions for MaskedOperatorWithMatrixArgumentOptions {
    fn use_mask_structure_of_stored_values_as_mask(&self) -> bool {
        self.use_mask_structure_of_stored_values_as_mask
    }

    fn use_mask_complement(&self) -> bool {
        self.use_mask_complement
    }
}

impl GetTransposeMatrixArgument for MaskedOperatorWithMatrixArgumentOptions {
    fn transpose_matrix_argument(&self) -> bool {
        self.transpose_matrix_argument
    }
}

impl GetGraphblasDescriptor for MaskedOperatorWithMatrixArgumentOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeMatrixArgument for MaskedOperatorWithMatrixArgumentOptions {
    fn with_negated_transpose_matrix_argument(&self) -> Self {
        MaskedOperatorWithMatrixArgumentOptions::new(
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
            MaskedOperatorWithMatrixArgumentOptions::new(
                self.clear_output_before_use,
                self.use_mask_structure_of_stored_values_as_mask,
                self.use_mask_complement,
                transpose_matrix,
            )
        }
    }
}

impl MaskedOperatorWithMatrixArgumentOptions {
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
                transpose_matrix_argument,
                false,
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
                transpose_matrix_argument,
                false,
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
        let default_options = MaskedOperatorWithMatrixArgumentOptions::new_default();
        let expected_value: GrB_Descriptor = ptr::null_mut();
        assert_eq!(default_options.graphblas_descriptor(), expected_value)
    }
}
