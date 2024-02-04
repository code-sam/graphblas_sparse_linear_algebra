use std::ptr;

use crate::graphblas_bindings::{
    GrB_DESC_C, GrB_DESC_CT0, GrB_DESC_CT0T1, GrB_DESC_CT1, GrB_DESC_R, GrB_DESC_RC, GrB_DESC_RCT0,
    GrB_DESC_RCT0T1, GrB_DESC_RCT1, GrB_DESC_RS, GrB_DESC_RSC, GrB_DESC_RSCT0, GrB_DESC_RSCT0T1,
    GrB_DESC_RSCT1, GrB_DESC_RST0, GrB_DESC_RST0T1, GrB_DESC_RST1, GrB_DESC_RT0, GrB_DESC_RT0T1,
    GrB_DESC_RT1, GrB_DESC_S, GrB_DESC_SC, GrB_DESC_SCT0, GrB_DESC_SCT0T1, GrB_DESC_SCT1,
    GrB_DESC_ST0, GrB_DESC_ST0T1, GrB_DESC_ST1, GrB_DESC_T0, GrB_DESC_T0T1, GrB_DESC_T1,
    GrB_Descriptor,
};

// pub enum GraphblasDescriptor {
//     // Default(*const GrB_Descriptor),
//     Default(GrB_Descriptor),
//     Value(GrB_Descriptor),
// }

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
    transpose_input0: bool,
    transpose_input1: bool,

    graphblas_descriptor: GrB_Descriptor,
}

impl OperatorOptions {
    pub fn new(
        clear_output_before_use: bool,
        use_mask_structure_of_stored_values_as_mask: bool,
        use_mask_complement: bool,
        transpose_input0: bool,
        transpose_input1: bool,
    ) -> Self {
        Self {
            clear_output_before_use,
            use_mask_structure_of_stored_values_as_mask,
            use_mask_complement,
            transpose_input0,
            transpose_input1,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                use_mask_structure_of_stored_values_as_mask,
                use_mask_complement,
                transpose_input0,
                transpose_input1,
            ),
        }
    }

    pub fn new_default() -> Self {
        let transpose_input0 = false;
        let transpose_input1 = false;
        let use_mask_complement = false;
        let use_mask_structure_of_stored_values_as_mask = false;
        let clear_output_before_use = false;

        Self {
            transpose_input0,
            transpose_input1,
            use_mask_complement,
            use_mask_structure_of_stored_values_as_mask,
            clear_output_before_use,

            graphblas_descriptor: graphblas_descriptor(
                clear_output_before_use,
                use_mask_structure_of_stored_values_as_mask,
                use_mask_complement,
                transpose_input0,
                transpose_input1,
            ),
        }
    }

    // pub fn graphblas_descriptor(&self) -> GrB_Descriptor {
    //     match (self.clearOutputBeforeUse, self.useMaskStructureOfStoredValuesAsMask, self.useMaskComplement, self.transposeInput0, self.transposeInput1) {
    //         (false,false,false,false,false) => unsafe {GraphblasDescriptor::Default(ptr::null())},
    //         (false,false,false,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_T1)},
    //         (false,false,false,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_T0)},
    //         (false,false,false,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_T0T1)},
    //         (false,false,true,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_C)},
    //         (false,false,true,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_CT1)},
    //         (false,false,true,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_CT0)},
    //         (false,false,true,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_CT0T1)},
    //         (false,true,false,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_S)},
    //         (false,true,false,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_ST1)},
    //         (false,true,false,true,false) => unsnewnsafe {GraphblasDescriptor::Value(GrB_DESC_SC)},
    //         (false,true,true,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_SCT0)},
    //         (false,true,true,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_SCT1)},
    //         (false,true,true,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_SCT0T1)},
    //         (true,false,false,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_R)},
    //         (true,false,false,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RT1)},
    //         (true,false,false,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RT0)},
    //         (true,false,false,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RT0T1)},
    //         (true,false,true,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RC)},
    //         (true,false,true,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RCT1)},
    //         (true,false,true,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RCT0)},
    //         (true,false,true,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RCT0T1)},
    //         (true,true,false,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RS)},
    //         (true,true,false,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RST1)},
    //         (true,true,false,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RST0)},
    //         (true,true,false,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RST0T1)},
    //         (true,true,true,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RSC)},
    //         (true,true,true,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RSCT0)},
    //         (true,true,true,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RSCT1)},
    //         (true,true,true,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RSCT0T1)},
    //     }
    // }

    // pub fn graphblas_descriptor(&self) -> GraphblasDescriptor {
    //     match (self.clearOutputBeforeUse, self.useMaskStructureOfStoredValuesAsMask, self.useMaskComplement, self.transposeInput0, self.transposeInput1) {
    //         (false,false,false,false,false) => unsafe {GraphblasDescriptor::Default(ptr::null())},
    //         (false,false,false,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_T1)},
    //         (false,false,false,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_T0)},
    //         (false,false,false,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_T0T1)},
    //         (false,false,true,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_C)},
    //         (false,false,true,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_CT1)},
    //         (false,false,true,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_CT0)},
    //         (false,false,true,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_CT0T1)},
    //         (false,true,false,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_S)},
    //         (false,true,false,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_ST1)},
    //         (false,true,false,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_ST0)},
    //         (false,true,false,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_ST0T1)},
    //         (false,true,true,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_SC)},
    //         (false,true,true,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_SCT0)},
    //         (false,true,true,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_SCT1)},
    //         (false,true,true,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_SCT0T1)},
    //         (true,false,false,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_R)},
    //         (true,false,false,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RT1)},
    //         (true,false,false,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RT0)},
    //         (true,false,false,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RT0T1)},
    //         (true,false,true,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RC)},
    //         (true,false,true,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RCT1)},
    //         (true,false,true,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RCT0)},
    //         (true,false,true,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RCT0T1)},
    //         (true,true,false,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RS)},
    //         (true,true,false,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RST1)},
    //         (true,true,false,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RST0)},
    //         (true,true,false,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RST0T1)},
    //         (true,true,true,false,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RSC)},
    //         (true,true,true,true,false) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RSCT0)},
    //         (true,true,true,false,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RSCT1)},
    //         (true,true,true,true,true) => unsafe {GraphblasDescriptor::Value(GrB_DESC_RSCT0T1)},
    //     }
    // }
}

pub trait GetGraphblasDescriptor {
    fn graphblas_descriptor(&self) -> GrB_Descriptor;
}

pub trait MutateOperatorOptions {
    fn with_negated_transpose_input0(&self) -> Self;
}

impl GetGraphblasDescriptor for OperatorOptions {
    fn graphblas_descriptor(&self) -> GrB_Descriptor {
        self.graphblas_descriptor
    }
}

impl MutateOperatorOptions for OperatorOptions {
    fn with_negated_transpose_input0(&self) -> Self {
        OperatorOptions::new(
            self.clear_output_before_use,
            self.use_mask_structure_of_stored_values_as_mask,
            self.use_mask_complement,
            !self.transpose_input0,
            self.transpose_input1,
        )
    }
}

fn graphblas_descriptor(
    clear_output_before_use: bool,
    use_mask_structure_of_stored_values_as_mask: bool,
    use_mask_complement: bool,
    transpose_input0: bool,
    transpose_input1: bool,
) -> GrB_Descriptor {
    match (
        clear_output_before_use,
        use_mask_structure_of_stored_values_as_mask,
        use_mask_complement,
        transpose_input0,
        transpose_input1,
    ) {
        (false, false, false, false, false) => ptr::null_mut(),
        (false, false, false, false, true) => unsafe { GrB_DESC_T1 },
        (false, false, false, true, false) => unsafe { GrB_DESC_T0 },
        (false, false, false, true, true) => unsafe { GrB_DESC_T0T1 },
        (false, false, true, false, false) => unsafe { GrB_DESC_C },
        (false, false, true, false, true) => unsafe { GrB_DESC_CT1 },
        (false, false, true, true, false) => unsafe { GrB_DESC_CT0 },
        (false, false, true, true, true) => unsafe { GrB_DESC_CT0T1 },
        (false, true, false, false, false) => unsafe { GrB_DESC_S },
        (false, true, false, false, true) => unsafe { GrB_DESC_ST1 },
        (false, true, false, true, false) => unsafe { GrB_DESC_ST0 },
        (false, true, false, true, true) => unsafe { GrB_DESC_ST0T1 },
        (false, true, true, false, false) => unsafe { GrB_DESC_SC },
        (false, true, true, true, false) => unsafe { GrB_DESC_SCT0 },
        (false, true, true, false, true) => unsafe { GrB_DESC_SCT1 },
        (false, true, true, true, true) => unsafe { GrB_DESC_SCT0T1 },
        (true, false, false, false, false) => unsafe { GrB_DESC_R },
        (true, false, false, false, true) => unsafe { GrB_DESC_RT1 },
        (true, false, false, true, false) => unsafe { GrB_DESC_RT0 },
        (true, false, false, true, true) => unsafe { GrB_DESC_RT0T1 },
        (true, false, true, false, false) => unsafe { GrB_DESC_RC },
        (true, false, true, false, true) => unsafe { GrB_DESC_RCT1 },
        (true, false, true, true, false) => unsafe { GrB_DESC_RCT0 },
        (true, false, true, true, true) => unsafe { GrB_DESC_RCT0T1 },
        (true, true, false, false, false) => unsafe { GrB_DESC_RS },
        (true, true, false, false, true) => unsafe { GrB_DESC_RST1 },
        (true, true, false, true, false) => unsafe { GrB_DESC_RST0 },
        (true, true, false, true, true) => unsafe { GrB_DESC_RST0T1 },
        (true, true, true, false, false) => unsafe { GrB_DESC_RSC },
        (true, true, true, true, false) => unsafe { GrB_DESC_RSCT0 },
        (true, true, true, false, true) => unsafe { GrB_DESC_RSCT1 },
        (true, true, true, true, true) => unsafe { GrB_DESC_RSCT0T1 },
        // (false, false, false, false, false) => unsafe {
        //     GraphblasDescriptor::Default(ptr::null_mut())
        // },
        // (false, false, false, false, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_T1)
        // },
        // (false, false, false, true, false) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_T0)
        // },
        // (false, false, false, true, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_T0T1)
        // },
        // (false, false, true, false, false) => unsafe { GraphblasDescriptor::Value(GrB_DESC_C) },
        // (false, false, true, false, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_CT1)
        // },
        // (false, false, true, true, false) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_CT0)
        // },
        // (false, false, true, true, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_CT0T1)
        // },
        // (false, true, false, false, false) => unsafe { GraphblasDescriptor::Value(GrB_DESC_S) },
        // (false, true, false, false, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_ST1)
        // },
        // (false, true, false, true, false) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_ST0)
        // },
        // (false, true, false, true, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_ST0T1)
        // },
        // (false, true, true, false, false) => unsafe { GraphblasDescriptor::Value(GrB_DESC_SC) },
        // (false, true, true, true, false) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_SCT0)
        // },
        // (false, true, true, false, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_SCT1)
        // },
        // (false, true, true, true, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_SCT0T1)
        // },
        // (true, false, false, false, false) => unsafe { GraphblasDescriptor::Value(GrB_DESC_R) },
        // (true, false, false, false, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RT1)
        // },
        // (true, false, false, true, false) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RT0)
        // },
        // (true, false, false, true, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RT0T1)
        // },
        // (true, false, true, false, false) => unsafe { GraphblasDescriptor::Value(GrB_DESC_RC) },
        // (true, false, true, false, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RCT1)
        // },
        // (true, false, true, true, false) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RCT0)
        // },
        // (true, false, true, true, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RCT0T1)
        // },
        // (true, true, false, false, false) => unsafe { GraphblasDescriptor::Value(GrB_DESC_RS) },
        // (true, true, false, false, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RST1)
        // },
        // (true, true, false, true, false) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RST0)
        // },
        // (true, true, false, true, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RST0T1)
        // },
        // (true, true, true, false, false) => unsafe { GraphblasDescriptor::Value(GrB_DESC_RSC) },
        // (true, true, true, true, false) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RSCT0)
        // },
        // (true, true, true, false, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RSCT1)
        // },newe, true) => unsafe {
        //     GraphblasDescriptor::Value(GrB_DESC_RSCT0T1)
        // },
    }
}

// pub enum ArithmeticOperatorOptionsList {
//     Default, // GrB_NULL
//     TransposeInput1, // T1
//     TransposeInput0, // T0
//     TransposeInput0And1, // T0T1,
//     UseMaskComplement, // C
//     UseMaskStructure, // use the masks's stored values as the mask
//     UseMaskStructureAndTransposeInput

// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options() {
        let default_options = OperatorOptions::new_default();
        let expected_value: GrB_Descriptor = ptr::null_mut();
        assert_eq!(default_options.graphblas_descriptor(), expected_value)
    }
}
