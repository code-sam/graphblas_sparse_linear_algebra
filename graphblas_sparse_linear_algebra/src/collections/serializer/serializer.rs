use suitesparse_graphblas_sys::GrB_Descriptor;

pub trait GetGraphblasSerializerDescriptor {
    unsafe fn graphblas_serializer_descriptor(&self) -> GrB_Descriptor;
    unsafe fn graphblas_serializer_descriptor_ref(&self) -> &GrB_Descriptor;
}
