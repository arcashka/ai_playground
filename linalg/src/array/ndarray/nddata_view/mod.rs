use super::{MemoryLayout, NDData};

pub struct NDDataView<'a, A, L: MemoryLayout, const N: usize, const M: usize> {
    base: &'a NDData<A, L, M>,
    dimensions: [usize; N],
    strides: [usize; N],
    offset: usize,
    _layout: std::marker::PhantomData<L>,
}
