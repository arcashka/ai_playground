use super::{ndstorage::Ownership, MemoryLayout, NDStorage};
use crate::Array;

use std::fmt;

impl<A, O, L, const N: usize> Array<NDStorage<A, O, L, N>>
where
    L: MemoryLayout,
    O: Ownership,
{
    pub fn get_element(&self, indices: [usize; N]) -> A {
        // TODO: add checks
        unsafe { self.storage.get(indices) }
    }
}

impl<A, O, L, const N: usize> fmt::Debug for Array<NDStorage<A, O, L, N>>
where
    A: fmt::Debug,
    O: Ownership,
    L: MemoryLayout,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("NDArray:\n{:?}", &self.storage))
    }
}
