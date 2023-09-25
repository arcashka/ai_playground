use crate::linear_regression;

pub trait ParametricAlgorithm<T>: linear_regression::LinearRegressionModel<T> {
    fn theta(&self) -> ndarray::ArrayView1<T>;
}
