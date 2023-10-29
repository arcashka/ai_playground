use crate::ndarray::NDArray2;
use crate::LinalgError;

impl<A> NDArray2<A> {
    pub fn push_row(&mut self, row: Vec<A>) -> Result<(), LinalgError>
    where
        A: Copy,
    {
        self.storage.push_row(row)
    }

    pub fn nrows(&self) -> usize {
        self.storage.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.storage.ncols()
    }
}
