use std::alloc::{alloc, dealloc, realloc, Layout};
use std::fmt::{self, Debug};
use std::ptr::write;

use crate::LinalgError;
use crate::Storage;

pub trait MemoryLayout {
    fn calculate_strides<const N: usize>(dimensions: [usize; N]) -> [usize; N];
}

pub struct RowMajor;
impl MemoryLayout for RowMajor {
    fn calculate_strides<const N: usize>(dimensions: [usize; N]) -> [usize; N] {
        let mut stride = 1;
        let mut strides = [0; N];
        for i in (0..N).rev() {
            strides[i] = stride;
            stride *= dimensions[i];
        }
        strides
    }
}

pub struct ColumnMajor;
impl MemoryLayout for ColumnMajor {
    fn calculate_strides<const N: usize>(dimensions: [usize; N]) -> [usize; N] {
        let mut stride = 1;
        let mut strides = [0; N];
        for i in 0..N {
            strides[i] = stride;
            stride *= dimensions[i];
        }
        strides
    }
}

pub struct NDStorage<A, L: MemoryLayout, const N: usize> {
    data: *mut A,
    dimensions: [usize; N],
    strides: [usize; N],
    _layout: std::marker::PhantomData<L>,
}

impl<A, L: MemoryLayout, const N: usize> Storage for NDStorage<A, L, N> {
    type Elem = A;
}

impl<A, L: MemoryLayout, const N: usize> Drop for NDStorage<A, L, N> {
    fn drop(&mut self) {
        let total_size = self.dimensions.iter().product();
        let layout = Layout::array::<A>(total_size).unwrap();
        unsafe { dealloc(self.data as *mut u8, layout) };
    }
}

impl<A, L: MemoryLayout, const N: usize> NDStorage<A, L, N> {
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
        NDStorage::<A, L, N> {
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

        Ok(NDStorage::<A, L, N> {
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

    fn update_dimensions(&mut self, new_dimensions: [usize; N]) {
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

impl<A: Clone, L: MemoryLayout, const N: usize> Clone for NDStorage<A, L, N> {
    fn clone(&self) -> Self {
        let total_size = self.dimensions.iter().product();
        let layout = Layout::array::<A>(total_size).unwrap();
        let new_data = unsafe { alloc(layout) as *mut A };
        for i in 0..total_size {
            let val = unsafe { self.data.add(i).read() };
            unsafe { new_data.add(i).write(val.clone()) };
        }

        NDStorage {
            data: new_data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
            _layout: std::marker::PhantomData,
        }
    }
}

impl<A: fmt::Debug, L: MemoryLayout, const N: usize> fmt::Debug for NDStorage<A, L, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("dimensions: {:?}\n", self.dimensions))?;
        f.write_fmt(format_args!("strides: {:?}\n", self.strides))?;
        f.write_fmt(format_args!("dimensions: {:?}\n", self.fmt_data()))
    }
}

impl<A: Copy> NDStorage<A, RowMajor, 2> {
    pub fn push_row(&mut self, row: Vec<A>) -> Result<(), LinalgError> {
        let dimensions = self.dimensions;
        let n_rows = dimensions[0];
        let n_cols = dimensions[1];

        if row.len() != n_cols {
            return Err(LinalgError::DimensionMismatch);
        }

        let new_n_rows = n_rows + 1;
        let total_size = n_rows * n_cols;
        let new_total_size = new_n_rows * n_cols;

        let old_layout = Layout::array::<A>(total_size).unwrap();
        let new_layout = Layout::array::<A>(new_total_size).unwrap();

        let new_data =
            unsafe { realloc(self.data as *mut _, old_layout, new_layout.size()) as *mut A };
        if new_data.is_null() {
            return Err(LinalgError::AllocationError);
        }

        for (i, &item) in row.iter().enumerate() {
            unsafe {
                write(new_data.add(total_size + i), item.clone());
            }
        }

        self.data = new_data;

        let mut dimensions = dimensions;
        dimensions[0] = new_n_rows;
        self.update_dimensions(dimensions);

        Ok(())
    }
}
