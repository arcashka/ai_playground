use super::{ColumnMajor, MemoryLayout, RowMajor};

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
