use super::{MemoryLayout, NDStorage, Ownership};
use crate::Storage;
use std::fmt;

impl<A, O, L, const N: usize> Storage for NDStorage<A, O, L, N>
where
    L: MemoryLayout,
    O: Ownership,
{
    type Elem = A;
}

impl<A, O, L, const N: usize> NDStorage<A, O, L, N>
where
    L: MemoryLayout,
    O: Ownership,
{
    pub fn strides(&self) -> &[usize; N] {
        &self.strides
    }

    pub fn dimensions(&self) -> &[usize; N] {
        &self.dimensions
    }

    pub unsafe fn get(&self, indices: [usize; N]) -> A {
        let mut offset = 0;
        for i in 0..N {
            offset += indices[i] * self.strides[i];
        }
        unsafe { self.data.get(offset) }
    }
}

impl<A, O, L, const N: usize> fmt::Debug for NDStorage<A, O, L, N>
where
    A: fmt::Debug,
    O: Ownership,
    L: MemoryLayout,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("dimensions: {:?}\n", self.dimensions))?;
        f.write_fmt(format_args!("strides: {:?}\n", self.strides))?;
        f.write_fmt(format_args!("data: {:?}\n", self.data))
    }
}
