mod ndarray_common_impl;
mod ndarray_impl;
pub mod ndstorage;
mod twodarray_impl;

pub use ndstorage::{ColumnMajor, MemoryLayout, NDStorage, Owning, RowMajor, View};

use super::Array;
pub type NDOwnedStorage<A, L, const N: usize> = NDStorage<A, Owning, L, N>;
pub type NDViewStorage<A, L, const N: usize> = NDStorage<A, View, L, N>;

pub type NDArray<A, L: MemoryLayout, const N: usize> = Array<NDOwnedStorage<A, L, N>>;
pub type NDArrayView<A, L: MemoryLayout, const N: usize> = Array<NDViewStorage<A, L, N>>;

pub type NDArray1<A> = Array<NDOwnedStorage<A, RowMajor, 1>>;
pub type NDArray2<A> = Array<NDOwnedStorage<A, RowMajor, 2>>;
pub type NDArrayView1<A> = Array<NDViewStorage<A, RowMajor, 1>>;
pub type NDArrayView2<A> = Array<NDViewStorage<A, RowMajor, 2>>;
