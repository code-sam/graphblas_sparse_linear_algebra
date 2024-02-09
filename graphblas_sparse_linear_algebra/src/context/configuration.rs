use suitesparse_graphblas_sys::{
    GxB_Format_Value, GxB_Format_Value_GxB_BY_COL, GxB_Format_Value_GxB_BY_ROW, GxB_Global_Option_set, GxB_Option_Field, GxB_Option_Field_GxB_FORMAT
};

use crate::error::SparseLinearAlgebraError;

use super::Context;

#[derive(Copy, Clone, Debug)]
pub enum MatrixStorageFormat {
    ByRow,
    ByColumn,
}

impl Into<GxB_Format_Value> for MatrixStorageFormat {
    fn into(self) -> GxB_Format_Value {
        match self {
            MatrixStorageFormat::ByRow => GxB_Format_Value_GxB_BY_ROW,
            MatrixStorageFormat::ByColumn => GxB_Format_Value_GxB_BY_COL,
        }
    }
}

pub(crate) trait SetMatrixFormat {
    fn set_matrix_format(
        &mut self,
        format: MatrixStorageFormat,
    ) -> Result<(), SparseLinearAlgebraError>;
}

impl SetMatrixFormat for Context {
    fn set_matrix_format(
        &mut self,
        format: MatrixStorageFormat,
    ) -> Result<(), SparseLinearAlgebraError> {
        self.call_without_detailed_error_information(|| unsafe {
            // TODO: use GrB_set() once it becomes available
            GxB_Global_Option_set(GxB_Option_Field_GxB_FORMAT, <MatrixStorageFormat as Into<GxB_Format_Value>>::into(format))
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

}
