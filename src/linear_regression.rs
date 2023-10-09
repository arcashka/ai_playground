use crate::array;

#[derive(Debug)]
pub enum LinearRegressionError {
    FailedCastToT,
    OperationFailed,
}

pub trait LinearRegressionModel<T> {
    fn predict(&self, x: &array::ArrayView1<T>) -> Result<T, LinearRegressionError>;
}
