use super::{MemoryLayout, NDData};

use crate::LinalgError;
use crate::Storage;

use std::alloc::{alloc, dealloc, Layout};
use std::fmt::{self, Debug};
use std::ptr::write;

impl<A, L: MemoryLayout, const N: usize> Storage for NDData<A, L, N> {
    type Elem = A;
}

impl<A, L: MemoryLayout, const N: usize> Drop for NDData<A, L, N> {
    fn drop(&mut self) {
        let total_size = self.dimensions.iter().product();
        let layout = Layout::array::<A>(total_size).unwrap();
        unsafe { dealloc(self.data as *mut u8, layout) };
    }
}

impl<A, L: MemoryLayout, const N: usize> NDData<A, L, N> {
    pub fn zeros(dimensions: [usize; N]) -> Self
    where
        A: num_traits::Zero,
    {
        let total_size = dimensions.iter().product();
        let layout = Layout::array::<A>(total_size).unwrap();
        let data = unsafe { alloc(layout) as *mut A };

        for i in 0..total_size {
            unsafe {
                write(data.add(i), A::zero());
            }
        }
        NDData::<A, L, N> {
            data,
            dimensions: dimensions.to_owned(),
            strides: L::calculate_strides(dimensions),
            _layout: std::marker::PhantomData,
        }
    }

    pub fn from_vec(vec: Vec<A>, dimensions: [usize; N]) -> Result<Self, LinalgError> {
        let total_size = dimensions.iter().product();
        if vec.len() != total_size {
            return Err(LinalgError::DimensionMismatch);
        }

        let boxed_slice = vec.into_boxed_slice();
        let data = Box::into_raw(boxed_slice) as *mut A;

        Ok(NDData::<A, L, N> {
            data,
            dimensions: dimensions.to_owned(),
            strides: L::calculate_strides(dimensions),
            _layout: std::marker::PhantomData,
        })
    }

    pub fn strides(&self) -> &[usize; N] {
        &self.strides
    }

    pub fn dimensions(&self) -> &[usize; N] {
        &self.dimensions
    }

    pub unsafe fn get(&self, indices: [usize; N]) -> A
    where
        A: Copy,
    {
        let mut offset = 0;
        for i in 0..N {
            offset += indices[i] * self.strides[i];
        }
        *self.data.add(offset)
    }

    pub(super) fn update_dimensions(&mut self, new_dimensions: [usize; N]) {
        self.dimensions = new_dimensions;
        self.strides = L::calculate_strides(self.dimensions);
    }

    fn fmt_data(&self) -> String
    where
        A: Debug,
    {
        let mut out = String::new();
        let size = self.dimensions().iter().product();
        for i in 0..size {
            unsafe {
                out += format!("{:?} ", *self.data.add(i)).as_str();
            }
        }
        out
    }
}

impl<A: Clone, L: MemoryLayout, const N: usize> Clone for NDData<A, L, N> {
    fn clone(&self) -> Self {
        let total_size = self.dimensions.iter().product();
        let layout = Layout::array::<A>(total_size).unwrap();
        let new_data = unsafe { alloc(layout) as *mut A };
        for i in 0..total_size {
            let val = unsafe { self.data.add(i).read() };
            unsafe { new_data.add(i).write(val.clone()) };
        }

        NDData {
            data: new_data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
            _layout: std::marker::PhantomData,
        }
    }
}

impl<A: fmt::Debug, L: MemoryLayout, const N: usize> fmt::Debug for NDData<A, L, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("dimensions: {:?}\n", self.dimensions))?;
        f.write_fmt(format_args!("strides: {:?}\n", self.strides))?;
        f.write_fmt(format_args!("dimensions: {:?}\n", self.fmt_data()))
    }
}
