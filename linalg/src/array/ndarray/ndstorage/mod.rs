mod common_data_impl;
mod common_storage_impl;
mod common_twodstorage_impl;
mod layout_impl;
mod owned_data_impl;
mod owned_storage_impl;
mod owned_twodstorage_impl;

pub trait MemoryLayout {
    fn calculate_strides<const N: usize>(dimensions: [usize; N]) -> [usize; N];
}

pub struct RowMajor;
pub struct ColumnMajor;

#[derive(Clone)]
pub struct NDStorage<A, O: Ownership, L: MemoryLayout, const N: usize> {
    data: Data<A, O>,
    dimensions: [usize; N],
    strides: [usize; N],
    layout: std::marker::PhantomData<L>,
}

pub trait Ownership {}
pub struct View;
pub struct Owning;
impl Ownership for View {}
impl Ownership for Owning {}

pub struct Data<A, O: Ownership> {
    data: *mut A,
    len: usize,
    ownership: std::marker::PhantomData<O>,
}
