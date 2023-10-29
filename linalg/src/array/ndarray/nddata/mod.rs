mod layout_impl;
mod ndstorage_impl;
mod twodstorage_impl;

pub trait MemoryLayout {
    fn calculate_strides<const N: usize>(dimensions: [usize; N]) -> [usize; N];
}

pub struct RowMajor;
pub struct ColumnMajor;

pub struct NDData<A, L: MemoryLayout, const N: usize> {
    data: *mut A,
    dimensions: [usize; N],
    strides: [usize; N],
    _layout: std::marker::PhantomData<L>,
}
