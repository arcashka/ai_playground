use super::{Data, MemoryLayout, NDStorage, Owning};
use crate::LinalgError;

impl<A, L: MemoryLayout, const N: usize> NDStorage<A, Owning, L, N> {
    pub fn zeros(dimensions: [usize; N]) -> Self
    where
        A: num_traits::Zero,
    {
        let total_size = dimensions.iter().product();
        NDStorage::<A, Owning, L, N> {
            data: Data::<A, Owning>::zeros(total_size),
            dimensions: dimensions.to_owned(),
            strides: L::calculate_strides(dimensions),
            layout: std::marker::PhantomData,
        }
    }

    pub fn from_vec(vec: Vec<A>, dimensions: [usize; N]) -> Result<Self, LinalgError> {
        let total_size = dimensions.iter().product();
        if vec.len() != total_size {
            return Err(LinalgError::DimensionMismatch);
        }

        let boxed_slice = vec.into_boxed_slice();
        let data = Box::into_raw(boxed_slice) as *mut A;

        Ok(NDStorage::<A, Owning, L, N> {
            data: Data::<A, Owning>::from_vec(vec),
            dimensions: dimensions.to_owned(),
            strides: L::calculate_strides(dimensions),
            layout: std::marker::PhantomData,
        })
    }

    pub(super) fn update_dimensions(&mut self, new_dimensions: [usize; N]) {
        self.dimensions = new_dimensions;
        self.strides = L::calculate_strides(self.dimensions);
    }
}
