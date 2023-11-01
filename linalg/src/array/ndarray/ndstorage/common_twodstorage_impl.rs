use super::{ColumnMajor, NDStorage, Ownership, RowMajor};

impl<A, O: Ownership> NDStorage<A, O, RowMajor, 2> {
    pub fn nrows(&self) -> usize {
        self.dimensions[0]
    }

    pub fn ncols(&self) -> usize {
        self.dimensions[1]
    }
}

impl<A, O: Ownership> NDStorage<A, O, ColumnMajor, 2> {
    pub fn nrows(&self) -> usize {
        self.dimensions[1]
    }

    pub fn ncols(&self) -> usize {
        self.dimensions[0]
    }
}
