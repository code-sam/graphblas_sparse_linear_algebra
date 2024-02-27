use suitesparse_graphblas_sys::GrB_Descriptor;

use super::{
    graphblas_descriptor, GetClearOutputBeforeUse, GetGraphblasDescriptor,
    GetMaskedOperatorOptions, GetOperatorMaskOptions, GetOperatorOptions, GetTransposeArguments,
    WithTransposeArguments,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for OperatorWithTransposableArgumentsOptions {}
unsafe impl Sync for OperatorWithTransposableArgumentsOptions {}

unsafe impl Send for MaskedOperatorWithTransposableArgumentsOptions {}
unsafe impl Sync for MaskedOperatorWithTransposableArgumentsOptions {}

#[derive(Debug, Clone)]
pub struct OperatorWithTransposableArgumentsOptions {
    clear_output_before_use: bool,
    transpose_first_argument: bool,
    transpose_second_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetOperatorWithTransposableArgumentsOptions:
    GetOperatorOptions + GetTransposeArguments
{
}

impl GetOperatorOptions for OperatorWithTransposableArgumentsOptions {}
impl GetOperatorWithTransposableArgumentsOptions for OperatorWithTransposableArgumentsOptions {}

impl GetClearOutputBeforeUse for OperatorWithTransposableArgumentsOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetTransposeArguments for OperatorWithTransposableArgumentsOptions {
    fn transpose_first_argument(&self) -> bool {
        self.transpose_first_argument
    }

    fn transpose_second_argument(&self) -> bool {
        self.transpose_second_argument
    }
}

impl GetGraphblasDescriptor for OperatorWithTransposableArgumentsOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeArguments for OperatorWithTransposableArgumentsOptions {
    fn with_negated_transpose_first_argument(&self) -> Self {
        OperatorWithTransposableArgumentsOptions::new(
            self.clear_output_before_use,
            !self.transpose_first_argument,
            self.transpose_second_argument,
        )
    }

    fn with_negated_transpose_second_argument(&self) -> Self {
        OperatorWithTransposableArgumentsOptions::new(
            self.clear_output_before_use,
            self.transpose_first_argument,
            !self.transpose_second_argument,
        )
    }

    fn with_transpose_first_argument(&self, transpose_first_argument: bool) -> Self {
        if transpose_first_argument == self.transpose_first_argument {
            self.to_owned()
        } else {
            OperatorWithTransposableArgumentsOptions::new(
                self.clear_output_before_use,
                transpose_first_argument,
                self.transpose_second_argument,
            )
        }
    }

    fn with_transpose_second_argument(&self, transpose_second_argument: bool) -> Self {
        if transpose_second_argument == self.transpose_second_argument {
            self.to_owned()
        } else {
            OperatorWithTransposableArgumentsOptions::new(
                self.clear_output_before_use,
                self.transpose_first_argument,
                transpose_second_argument,
            )
        }
    }

    fn with_transpose_matrix_arguments(
        &self,
        transpose_first_argument: bool,
        transpose_second_argument: bool,
    ) -> Self {
        if transpose_first_argument == self.transpose_first_argument
            && transpose_second_argument == self.transpose_second_argument
        {
            self.to_owned()
        } else {
            OperatorWithTransposableArgumentsOptions::new(
                self.clear_output_before_use,
                transpose_first_argument,
                transpose_second_argument,
            )
        }
    }
}

impl OperatorWithTransposableArgumentsOptions {
    pub fn new(
        clear_output_before_use: bool,
        transpose_first_argument: bool,
        transpose_second_argument: bool,
    ) -> Self {
        Self {
            clear_output_before_use,
            transpose_first_argument,
            transpose_second_argument,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                transpose_first_argument,
                transpose_second_argument,
            ),
        }
    }

    pub fn new_default() -> Self {
        let transpose_first_argument = false;
        let transpose_second_argument = false;
        let clear_output_before_use = false;

        Self {
            transpose_first_argument,
            transpose_second_argument,
            clear_output_before_use,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                transpose_first_argument,
                transpose_second_argument,
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaskedOperatorWithTransposableArgumentsOptions {
    clear_output_before_use: bool,
    use_mask_structure_of_stored_values_as_mask: bool,
    use_mask_complement: bool,
    transpose_first_argument: bool,
    transpose_second_argument: bool,

    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetMaskedOperatorWithTransposableArgumentsOptions:
    GetMaskedOperatorOptions + GetTransposeArguments
{
}

impl GetMaskedOperatorOptions for MaskedOperatorWithTransposableArgumentsOptions {}
impl GetMaskedOperatorWithTransposableArgumentsOptions
    for MaskedOperatorWithTransposableArgumentsOptions
{
}

impl GetClearOutputBeforeUse for MaskedOperatorWithTransposableArgumentsOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetOperatorMaskOptions for MaskedOperatorWithTransposableArgumentsOptions {
    fn use_mask_structure_of_stored_values_as_mask(&self) -> bool {
        self.use_mask_structure_of_stored_values_as_mask
    }

    fn use_mask_complement(&self) -> bool {
        self.use_mask_complement
    }
}

impl GetTransposeArguments for MaskedOperatorWithTransposableArgumentsOptions {
    fn transpose_first_argument(&self) -> bool {
        self.transpose_first_argument
    }

    fn transpose_second_argument(&self) -> bool {
        self.transpose_second_argument
    }
}

impl GetGraphblasDescriptor for MaskedOperatorWithTransposableArgumentsOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl WithTransposeArguments for MaskedOperatorWithTransposableArgumentsOptions {
    fn with_negated_transpose_first_argument(&self) -> Self {
        MaskedOperatorWithTransposableArgumentsOptions::new(
            self.clear_output_before_use,
            self.use_mask_structure_of_stored_values_as_mask,
            self.use_mask_complement,
            !self.transpose_first_argument,
            self.transpose_second_argument,
        )
    }

    fn with_negated_transpose_second_argument(&self) -> Self {
        MaskedOperatorWithTransposableArgumentsOptions::new(
            self.clear_output_before_use,
            self.use_mask_structure_of_stored_values_as_mask,
            self.use_mask_complement,
            self.transpose_first_argument,
            !self.transpose_second_argument,
        )
    }

    fn with_transpose_first_argument(&self, transpose_first_argument: bool) -> Self {
        if transpose_first_argument == self.transpose_first_argument {
            self.to_owned()
        } else {
            MaskedOperatorWithTransposableArgumentsOptions::new(
                self.clear_output_before_use,
                self.use_mask_structure_of_stored_values_as_mask,
                self.use_mask_complement,
                transpose_first_argument,
                self.transpose_second_argument,
            )
        }
    }

    fn with_transpose_second_argument(&self, transpose_second_argument: bool) -> Self {
        if transpose_second_argument == self.transpose_second_argument {
            self.to_owned()
        } else {
            MaskedOperatorWithTransposableArgumentsOptions::new(
                self.clear_output_before_use,
                self.use_mask_structure_of_stored_values_as_mask,
                self.use_mask_complement,
                self.transpose_first_argument,
                transpose_second_argument,
            )
        }
    }

    fn with_transpose_matrix_arguments(
        &self,
        transpose_first_argument: bool,
        transpose_second_argument: bool,
    ) -> Self {
        if transpose_first_argument == self.transpose_first_argument
            && transpose_second_argument == self.transpose_second_argument
        {
            self.to_owned()
        } else {
            MaskedOperatorWithTransposableArgumentsOptions::new(
                self.clear_output_before_use,
                self.use_mask_structure_of_stored_values_as_mask,
                self.use_mask_complement,
                transpose_first_argument,
                transpose_second_argument,
            )
        }
    }
}

impl MaskedOperatorWithTransposableArgumentsOptions {
    pub fn new(
        clear_output_before_use: bool,
        use_mask_structure_of_stored_values_as_mask: bool,
        use_mask_complement: bool,
        transpose_first_argument: bool,
        transpose_second_argument: bool,
    ) -> Self {
        Self {
            clear_output_before_use,
            use_mask_structure_of_stored_values_as_mask,
            use_mask_complement,
            transpose_first_argument,
            transpose_second_argument,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                transpose_first_argument,
                transpose_second_argument,
            ),
        }
    }

    pub fn new_default() -> Self {
        let use_mask_structure_of_stored_values_as_mask = false;
        let use_mask_complement = false;
        let transpose_first_argument = false;
        let transpose_second_argument = false;
        let clear_output_before_use = false;

        Self {
            clear_output_before_use,
            use_mask_structure_of_stored_values_as_mask,
            use_mask_complement,
            transpose_first_argument,
            transpose_second_argument,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                false,
                false,
                transpose_first_argument,
                transpose_second_argument,
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
        let default_options = MaskedOperatorWithTransposableArgumentsOptions::new_default();
        let expected_value: GrB_Descriptor = ptr::null_mut();
        assert_eq!(default_options.graphblas_descriptor(), expected_value)
    }
}
