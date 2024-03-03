use suitesparse_graphblas_sys::GrB_Descriptor;

use super::{
    graphblas_descriptor, GetClearOutputBeforeUse, GetGraphblasDescriptor, GetOperatorMaskOptions,
};

// Implemented methods do not provide mutable access to GraphBLAS operators or options.
// Code review must consider that no mtable access is provided.
// https://doc.rust-lang.org/nomicon/send-and-sync.html
unsafe impl Send for OperatorOptions {}
unsafe impl Sync for OperatorOptions {}

#[derive(Debug, Clone)]
pub struct OperatorOptions {
    clear_output_before_use: bool,
    use_mask_structure_of_stored_values_as_mask: bool,
    use_mask_complement: bool,
    graphblas_descriptor: GrB_Descriptor,
}

pub trait GetOperatorOptions:
    GetClearOutputBeforeUse + GetOperatorMaskOptions + GetGraphblasDescriptor
{
}

impl GetOperatorOptions for OperatorOptions {}

impl GetClearOutputBeforeUse for OperatorOptions {
    fn clear_output_before_use(&self) -> bool {
        self.clear_output_before_use
    }
}

impl GetOperatorMaskOptions for OperatorOptions {
    fn use_mask_structure_of_stored_values_as_mask(&self) -> bool {
        self.use_mask_structure_of_stored_values_as_mask
    }

    fn use_mask_complement(&self) -> bool {
        self.use_mask_complement
    }
}

impl GetGraphblasDescriptor for OperatorOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl OperatorOptions {
    pub fn new(
        clear_output_before_use: bool,
        use_mask_structure_of_stored_values_as_mask: bool,
        use_mask_complement: bool,
    ) -> Self {
        Self {
            clear_output_before_use,
            use_mask_structure_of_stored_values_as_mask,
            use_mask_complement,
            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                use_mask_structure_of_stored_values_as_mask,
                use_mask_complement,
                false,
                false,
            ),
        }
    }

    pub fn new_default() -> Self {
        let clear_output_before_use = false;
        let use_mask_structure_of_stored_values_as_mask = false;
        let use_mask_complement = false;
        Self {
            clear_output_before_use,
            use_mask_structure_of_stored_values_as_mask,
            use_mask_complement,
            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                use_mask_structure_of_stored_values_as_mask,
                use_mask_complement,
                false,
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
        let default_options = OperatorOptions::new_default();
        let expected_value: GrB_Descriptor = ptr::null_mut();
        assert_eq!(default_options.graphblas_descriptor(), expected_value)
    }
}
