use std::ptr;

use suitesparse_graphblas_sys::{
    GrB_DESC_C, GrB_DESC_CT0, GrB_DESC_CT0T1, GrB_DESC_CT1, GrB_DESC_R, GrB_DESC_RC, GrB_DESC_RCT0,
    GrB_DESC_RCT0T1, GrB_DESC_RCT1, GrB_DESC_RS, GrB_DESC_RSC, GrB_DESC_RSCT0, GrB_DESC_RSCT0T1,
    GrB_DESC_RSCT1, GrB_DESC_RST0, GrB_DESC_RST0T1, GrB_DESC_RST1, GrB_DESC_RT0, GrB_DESC_RT0T1,
    GrB_DESC_RT1, GrB_DESC_S, GrB_DESC_SC, GrB_DESC_SCT0, GrB_DESC_SCT0T1, GrB_DESC_SCT1,
    GrB_DESC_ST0, GrB_DESC_ST0T1, GrB_DESC_ST1, GrB_DESC_T0, GrB_DESC_T0T1, GrB_DESC_T1,
    GrB_Descriptor,
};

pub trait GetGraphblasDescriptor {
    fn graphblas_descriptor(&self) -> GrB_Descriptor;
}

pub(crate) fn graphblas_descriptor(
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
    }
}
