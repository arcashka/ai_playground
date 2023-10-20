use std::alloc::{alloc, dealloc, Layout};
use std::ptr::write;

pub enum LayoutType {
    RowMajor,
    ColumnMajor,
}

pub trait MemoryLayout {
    type NValues;
    fn new(dimensions: &Self::NValues) -> Self;
    fn strides(&self) -> &Self::NValues;
    fn dimensions(&self) -> &Self::NValues;
}

pub struct NDArrayStorage<const L: LayoutType, const N: usize> {
    dimensions: [usize; N],
    strides: [usize; N],
}

#[inline]
fn calculate_strides<const L: LayoutType, const N: usize>(dimensions: &[usize; N]) {
    let mut stride = 1;
    let mut strides = [0; N];
    match L {
        LayoutType::RowMajor => {
            for i in (0..N).rev() {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        }
        LayoutType::ColumnMajor => {
            for i in 0..N {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        }
    }
    strides
}

impl<const L: LayoutType, const N: usize> NDArrayStorage<L, N> {
    pub fn update_dimensions(&mut self, dimensions: &[usize; N]) {
        self.dimensions = dimensions;
        self.strides = calculate_strides(dimensions);
    }
}

impl<const L: LayoutType, const N: usize> MemoryLayout for NDArrayStorage<L, N> {
    type NValues = [usize; N];
    fn new(dimensions: &Self::NValues) -> Self {
        NDArrayStorage::<L, N>{
            dimensions,
            strides: calculate_strides(dimensions),
        }
    }

    fn strides(&self) -> &Self::NValues {
        &self.strides
    }

    fn dimensions(&self) -> &Self::NValues {
        &self.dimensions
    }
}

pub struct NDArray<A, L: MemoryLayout>
{
    data: *mut A,
    layout: L,
}

impl<A, L> NDArray<A, L>
where
    L: MemoryLayout,
{
    pub fn new(dim: L::NValues, initial: A) -> Self
    where
        A: Clone,
    {
        let total_size = dim.iter().product();

        let layout = Layout::array::<A>(total_size).unwrap();
        let data = unsafe { alloc(layout) as *mut A };

        for i in 0..total_size {
            unsafe {
                write(data.add(i), initial.clone());
            }
        }

        let strides = L::compute_strides(&dim);
        let layout
        NDArray { data, dim, strides }
    }
}

impl<A, const N: usize> Drop for NDArray<A, N> {
    fn drop(&mut self) {
        let total_size = self.dim.iter().product();
        let layout = Layout::array::<A>(total_size).unwrap();
        unsafe { dealloc(self.data as *mut u8, layout) };
    }
}
