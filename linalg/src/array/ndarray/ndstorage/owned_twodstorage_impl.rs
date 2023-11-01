use super::{NDStorage, Owning, RowMajor};

use crate::LinalgError;

impl<A> NDStorage<A, Owning, RowMajor, 2> {
    pub fn push_row(&mut self, row: Vec<A>) -> Result<(), LinalgError>
    where
        A: Copy,
    {
        let n_rows = self.dimensions[0];
        let n_cols = self.dimensions[1];

        if row.len() != n_cols {
            return Err(LinalgError::DimensionMismatch);
        }

        self.data.append(row)?;
        self.update_dimensions([n_rows + 1, n_cols]);
        Ok(())
    }
}
