#[derive(Debug)]
pub enum LinearRegressionError {
    FailedCastToT,
    OperationFailed,
}

pub trait LinearRegressionModel<T> {
    fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError>;
}
