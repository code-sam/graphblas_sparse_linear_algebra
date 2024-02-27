pub trait GetClearOutputBeforeUse {
    fn clear_output_before_use(&self) -> bool;
}

pub trait GetTransposeArguments {
    fn transpose_first_argument(&self) -> bool;
    fn transpose_second_argument(&self) -> bool;
}

pub trait GetTransposeMatrixArgument {
    fn transpose_matrix_argument(&self) -> bool;
}

pub trait GetTransposeFirstMatrixArgument {
    fn transpose_first_matrix_argument(&self) -> bool;
}

pub trait GetTransposeSecondMatrixArgument {
    fn transpose_second_matrix_argument(&self) -> bool;
}

pub trait GetOperatorMaskOptions {
    fn use_mask_structure_of_stored_values_as_mask(&self) -> bool;
    fn use_mask_complement(&self) -> bool;
}

pub trait WithTransposeMatrixArgument {
    fn with_negated_transpose_matrix_argument(&self) -> Self;
    fn with_transpose_matrix_argument(&self, transpose_matrix_argument: bool) -> Self;
}

pub trait WithTransposeArguments {
    fn with_negated_transpose_first_argument(&self) -> Self;
    fn with_negated_transpose_second_argument(&self) -> Self;

    fn with_transpose_first_argument(&self, transpose_matrix_argument: bool) -> Self;
    fn with_transpose_second_argument(&self, transpose_matrix_argument: bool) -> Self;

    fn with_transpose_matrix_arguments(
        &self,
        transpose_first_matrix_argument: bool,
        transpose_second_matrix_argument: bool,
    ) -> Self;
}
