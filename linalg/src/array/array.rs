use crate::array::Storage;

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
