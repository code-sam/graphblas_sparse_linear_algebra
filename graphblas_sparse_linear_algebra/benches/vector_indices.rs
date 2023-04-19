use graphblas_sparse_linear_algebra::collections::sparse_vector::GetElementIndices;
use graphblas_sparse_linear_algebra::collections::sparse_vector::GetVectorElementList;
use graphblas_sparse_linear_algebra::collections::sparse_vector::SetVectorElement;
use graphblas_sparse_linear_algebra::collections::sparse_vector::SparseVectorTrait;
use graphblas_sparse_linear_algebra::collections::sparse_vector::{SparseVector, VectorElement};
use graphblas_sparse_linear_algebra::context::{Context, Mode};
use graphblas_sparse_linear_algebra::index::ElementIndex;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;

fn bench_vector_indices(c: &mut Criterion) {
    let context = Context::init_ready(Mode::NonBlocking).unwrap();
    let mut rng = rand::thread_rng();

    let mut vector = SparseVector::<ElementIndex>::new(&context, &100000).unwrap();
    let fill_factor = 0.95;

    for i in 0..vector.length().unwrap() {
        let s: f64 = rng.gen();
        if s <= fill_factor {
            vector
                .set_element(VectorElement::new(i as ElementIndex, i))
                .unwrap();
        }
    }

    // c.bench_function("vector indices from iterator", |b| {
    //     b.iter(|| bench_index_iterator(vector.clone()))
    // });

    c.bench_function("vector indices from element list", |b| {
        b.iter(|| bench_element_list(vector.clone()))
    });
}

criterion_group!(benches, bench_vector_indices);
criterion_main!(benches);

// fn bench_index_iterator(vector: SparseVector<ElementIndex>) {
//     let indices = vector.indices().unwrap();
//     // println!("{:?}", indices)
// }

fn bench_element_list(vector: SparseVector<ElementIndex>) {
    let indices = vector.element_indices().unwrap();
    // println!("{:?}", indices)
}