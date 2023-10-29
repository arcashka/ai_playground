pub mod ndarray;
pub mod storage_trait;

pub use storage_trait::Storage;

#[derive(Debug)]
pub enum LinalgError {
    DimensionMismatch,
    AllocationError,
}

#[derive(Clone)]
pub struct Array<S>
where
    S: Storage,
{
    pub(crate) storage: S,
}
