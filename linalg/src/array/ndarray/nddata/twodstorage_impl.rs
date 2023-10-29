use super::{ColumnMajor, NDData, RowMajor};

use std::alloc::{realloc, Layout};
use std::ptr::write;

use crate::LinalgError;

impl<A> NDData<A, RowMajor, 2> {
    pub fn push_row(&mut self, row: Vec<A>) -> Result<(), LinalgError>
    where
        A: Copy,
    {
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

    pub fn nrows(&self) -> usize {
        self.dimensions[0]
    }

    pub fn ncols(&self) -> usize {
        self.dimensions[1]
    }
}

impl<A> NDData<A, ColumnMajor, 2> {
    pub fn nrows(&self) -> usize {
        self.dimensions[1]
    }

    pub fn ncols(&self) -> usize {
        self.dimensions[0]
    }
}
