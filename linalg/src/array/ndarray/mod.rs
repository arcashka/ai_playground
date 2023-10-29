pub mod ndarray_impl;
pub mod nddata;
pub mod nddata_view;
pub mod ndstorage;
pub mod twodarray_impl;

pub use nddata::{ColumnMajor, MemoryLayout, NDData, RowMajor};

use super::Array;
pub type NDArray<A, const N: usize> = Array<NDData<A, RowMajor, N>>;
pub type NDArray1<A> = Array<NDData<A, RowMajor, 1>>;
pub type NDArray2<A> = Array<NDData<A, RowMajor, 2>>;

pub trait RawData {
    type Elem;
}
