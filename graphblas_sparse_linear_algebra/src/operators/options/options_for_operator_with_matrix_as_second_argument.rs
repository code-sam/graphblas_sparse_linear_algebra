use suitesparse_graphblas_sys::GrB_Descriptor;

use super::{
    graphblas_descriptor, GetClearOutputBeforeUse, GetGraphblasDescriptor,
    GetMaskedOperatorOptions, GetOperatorMaskOptions, GetOperatorOptions,
    GetTransposeSecondMatrixArgument, WithTransposeMatrixArgument,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for OptionsForOperatorWithMatrixAsSecondArgument {}
unsafe impl Sync for OptionsForOperatorWithMatrixAsSecondArgument {}

unsafe impl Send for OptionsForMaskedOperatorWithMatrixAsSecondArgument {}
unsafe impl Sync for OptionsForMaskedOperatorWithMatrixAsSecondArgument {}

#[derive(Debug, Clone)]
pub struct OptionsForOperatorWithMatrixAsSecondArgument {
    clear_output_before_use: bool,
    transpose_matrix_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetOptionsForOperatorWithMatrixAsSecondArgument:
    GetOperatorOptions + GetTransposeSecondMatrixArgument
{
}

impl GetOperatorOptions for OptionsForOperatorWithMatrixAsSecondArgument {}
impl GetOptionsForOperatorWithMatrixAsSecondArgument
    for OptionsForOperatorWithMatrixAsSecondArgument
{
}

impl GetClearOutputBeforeUse for OptionsForOperatorWithMatrixAsSecondArgument {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetTransposeSecondMatrixArgument for OptionsForOperatorWithMatrixAsSecondArgument {
    fn transpose_second_matrix_argument(&self) -> bool {
        self.transpose_matrix_argument
    }
}

impl GetGraphblasDescriptor for OptionsForOperatorWithMatrixAsSecondArgument {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeMatrixArgument for OptionsForOperatorWithMatrixAsSecondArgument {
    fn with_negated_transpose_matrix_argument(&self) -> Self {
        OptionsForOperatorWithMatrixAsSecondArgument::new(
            self.clear_output_before_use,
            !self.transpose_matrix_argument,
        )
    }

    fn with_transpose_matrix_argument(&self, transpose_matrix_argument: bool) -> Self {
        if transpose_matrix_argument == self.transpose_matrix_argument {
            self.to_owned()
        } else {
            OptionsForOperatorWithMatrixAsSecondArgument::new(
                self.clear_output_before_use,
                transpose_matrix_argument,
            )
        }
    }
}

impl OptionsForOperatorWithMatrixAsSecondArgument {
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
pub struct OptionsForMaskedOperatorWithMatrixAsSecondArgument {
    clear_output_before_use: bool,
    use_mask_structure_of_stored_values_as_mask: bool,
    use_mask_complement: bool,
    transpose_matrix_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetOptionsForMaskedOperatorWithMatrixAsSecondArgument:
    GetMaskedOperatorOptions + GetTransposeSecondMatrixArgument
{
}

impl GetMaskedOperatorOptions for OptionsForMaskedOperatorWithMatrixAsSecondArgument {}
impl GetOptionsForMaskedOperatorWithMatrixAsSecondArgument
    for OptionsForMaskedOperatorWithMatrixAsSecondArgument
{
}

impl GetClearOutputBeforeUse for OptionsForMaskedOperatorWithMatrixAsSecondArgument {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetOperatorMaskOptions for OptionsForMaskedOperatorWithMatrixAsSecondArgument {
    fn use_mask_structure_of_stored_values_as_mask(&self) -> bool {
        self.use_mask_structure_of_stored_values_as_mask
    }

    fn use_mask_complement(&self) -> bool {
        self.use_mask_complement
    }
}

impl GetTransposeSecondMatrixArgument for OptionsForMaskedOperatorWithMatrixAsSecondArgument {
    fn transpose_second_matrix_argument(&self) -> bool {
        self.transpose_matrix_argument
    }
}

impl GetGraphblasDescriptor for OptionsForMaskedOperatorWithMatrixAsSecondArgument {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeMatrixArgument for OptionsForMaskedOperatorWithMatrixAsSecondArgument {
    fn with_negated_transpose_matrix_argument(&self) -> Self {
        OptionsForMaskedOperatorWithMatrixAsSecondArgument::new(
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
            OptionsForMaskedOperatorWithMatrixAsSecondArgument::new(
                self.clear_output_before_use,
                self.use_mask_structure_of_stored_values_as_mask,
                self.use_mask_complement,
                transpose_matrix,
            )
        }
    }
}

impl OptionsForMaskedOperatorWithMatrixAsSecondArgument {
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
        let default_options = OptionsForMaskedOperatorWithMatrixAsSecondArgument::new_default();
        let expected_value: GrB_Descriptor = ptr::null_mut();
        assert_eq!(default_options.graphblas_descriptor(), expected_value)
    }
}
