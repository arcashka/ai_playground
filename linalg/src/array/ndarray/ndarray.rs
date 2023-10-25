use crate::array::ndarray::*;
use crate::Array;
use crate::LinalgError;

use std::fmt;

pub type NDArray<A, const N: usize> = Array<ndstorage::NDStorage<A, ndstorage::RowMajor, N>>;
pub type NDArray1<A> = Array<ndstorage::NDStorage<A, ndstorage::RowMajor, 1>>;
pub type NDArray2<A> = Array<ndstorage::NDStorage<A, ndstorage::RowMajor, 2>>;

impl<A, L, const N: usize> Array<NDStorage<A, L, N>>
where
    L: MemoryLayout,
{
    pub fn zeros(dimensions: [usize; N]) -> Self
    where
        A: num_traits::Zero,
    {
        Self {
            storage: NDStorage::<A, L, N>::zeros(dimensions),
        }
    }

    pub fn from_vec(vec: Vec<A>, dimensions: [usize; N]) -> Result<Self, LinalgError>
    where
        A: Copy,
    {
        Ok(Self {
            storage: NDStorage::<A, L, N>::from_vec(vec, dimensions)?,
        })
    }

    pub fn get_element(&self, indices: [usize; N]) -> A
    where
        A: Copy,
    {
        unsafe { self.storage.get(indices) }
    }
}

impl<A> Array<NDStorage<A, RowMajor, 2>>
where
    A: Copy,
{
    pub fn push_row(&mut self, row: Vec<A>) -> Result<(), LinalgError> {
        self.storage.push_row(row)
    }
}

impl<A: fmt::Debug, L: MemoryLayout, const N: usize> fmt::Debug for Array<NDStorage<A, L, N>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("NDArray:\n{:?}", &self.storage))
    }
}

#[cfg(test)]
mod tests {
    use super::ndstorage::*;
    use super::*;

    #[test]
    fn create_zeros_array() {
        let a = Array::<NDStorage<f64, RowMajor, 3>>::zeros([3, 3, 7]);
        assert_eq!(*a.storage.strides(), [21, 7, 1]);
        assert_eq!(a.get_element([0, 0, 4]), 0.0);

        let b = Array::<NDStorage<f64, ColumnMajor, 3>>::zeros([3, 3, 7]);
        assert_eq!(*b.storage.strides(), [1, 3, 9]);
    }

    #[test]
    fn create_array_from_vec() {
        let v = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let a = Array::<NDStorage<f64, RowMajor, 2>>::from_vec(v, [1, 7]).unwrap();
        assert_eq!(*a.storage.strides(), [7, 1]);
        assert_eq!(a.get_element([0, 4]), 0.6);

        let v2 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let a2 = Array::<NDStorage<f64, ColumnMajor, 3>>::from_vec(v2, [3, 2, 2]).unwrap();
        assert_eq!(*a2.storage.strides(), [1, 3, 6]);
        assert_eq!(a2.get_element([0, 1, 1]), 1.0);
    }

    #[test]
    fn push_row() {
        let v = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let mut a = Array::<NDStorage<f64, RowMajor, 2>>::from_vec(v, [3, 4]).unwrap();
        let result = a.push_row(vec![13., 14., 15., 16.]);
        result.expect("should be fine");
        assert_eq!(*a.storage.strides(), [4, 1]);
        assert_eq!(*a.storage.dimensions(), [4, 4]);
        assert_eq!(a.get_element([3, 2]), 15.);
    }
}
