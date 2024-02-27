use suitesparse_graphblas_sys::GrB_Descriptor;

use super::{
    graphblas_descriptor, GetClearOutputBeforeUse, GetGraphblasDescriptor,
    GetMaskedOperatorOptions, GetOperatorMaskOptions, GetOperatorOptions,
    GetTransposeFirstMatrixArgument, GetTransposeMatrixArgument, WithTransposeMatrixArgument,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for OperatorWithMatrixAsFirstArgumentOptions {}
unsafe impl Sync for OperatorWithMatrixAsFirstArgumentOptions {}

unsafe impl Send for MaskedOperatorWithMatrixAsFirstArgumentOptions {}
unsafe impl Sync for MaskedOperatorWithMatrixAsFirstArgumentOptions {}

#[derive(Debug, Clone)]
pub struct OperatorWithMatrixAsFirstArgumentOptions {
    clear_output_before_use: bool,
    transpose_matrix_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetOperatorWithMatrixAsFirstArgumentOptions:
    GetOperatorOptions + GetTransposeFirstMatrixArgument
{
}

impl GetOperatorOptions for OperatorWithMatrixAsFirstArgumentOptions {}
impl GetOperatorWithMatrixAsFirstArgumentOptions for OperatorWithMatrixAsFirstArgumentOptions {}

impl GetClearOutputBeforeUse for OperatorWithMatrixAsFirstArgumentOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetTransposeFirstMatrixArgument for OperatorWithMatrixAsFirstArgumentOptions {
    fn transpose_first_matrix_argument(&self) -> bool {
        self.transpose_matrix_argument
    }
}

impl GetGraphblasDescriptor for OperatorWithMatrixAsFirstArgumentOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeMatrixArgument for OperatorWithMatrixAsFirstArgumentOptions {
    fn with_negated_transpose_matrix_argument(&self) -> Self {
        OperatorWithMatrixAsFirstArgumentOptions::new(
            self.clear_output_before_use,
            !self.transpose_matrix_argument,
        )
    }

    fn with_transpose_matrix_argument(&self, transpose_matrix_argument: bool) -> Self {
        if transpose_matrix_argument == self.transpose_matrix_argument {
            self.to_owned()
        } else {
            OperatorWithMatrixAsFirstArgumentOptions::new(
                self.clear_output_before_use,
                transpose_matrix_argument,
            )
        }
    }
}

impl OperatorWithMatrixAsFirstArgumentOptions {
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
pub struct MaskedOperatorWithMatrixAsFirstArgumentOptions {
    clear_output_before_use: bool,
    use_mask_structure_of_stored_values_as_mask: bool,
    use_mask_complement: bool,
    transpose_matrix_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetMaskedOperatorWithMatrixAsFirstArgumentOptions:
    GetMaskedOperatorOptions + GetTransposeFirstMatrixArgument
{
}

impl GetMaskedOperatorOptions for MaskedOperatorWithMatrixAsFirstArgumentOptions {}
impl GetMaskedOperatorWithMatrixAsFirstArgumentOptions
    for MaskedOperatorWithMatrixAsFirstArgumentOptions
{
}

impl GetClearOutputBeforeUse for MaskedOperatorWithMatrixAsFirstArgumentOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetOperatorMaskOptions for MaskedOperatorWithMatrixAsFirstArgumentOptions {
    fn use_mask_structure_of_stored_values_as_mask(&self) -> bool {
        self.use_mask_structure_of_stored_values_as_mask
    }

    fn use_mask_complement(&self) -> bool {
        self.use_mask_complement
    }
}

impl GetTransposeFirstMatrixArgument for MaskedOperatorWithMatrixAsFirstArgumentOptions {
    fn transpose_first_matrix_argument(&self) -> bool {
        self.transpose_matrix_argument
    }
}

impl GetGraphblasDescriptor for MaskedOperatorWithMatrixAsFirstArgumentOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeMatrixArgument for MaskedOperatorWithMatrixAsFirstArgumentOptions {
    fn with_negated_transpose_matrix_argument(&self) -> Self {
        MaskedOperatorWithMatrixAsFirstArgumentOptions::new(
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
            MaskedOperatorWithMatrixAsFirstArgumentOptions::new(
                self.clear_output_before_use,
                self.use_mask_structure_of_stored_values_as_mask,
                self.use_mask_complement,
                transpose_matrix,
            )
        }
    }
}

impl MaskedOperatorWithMatrixAsFirstArgumentOptions {
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
        let default_options = MaskedOperatorWithMatrixAsFirstArgumentOptions::new_default();
        let expected_value: GrB_Descriptor = ptr::null_mut();
        assert_eq!(default_options.graphblas_descriptor(), expected_value)
    }
}
