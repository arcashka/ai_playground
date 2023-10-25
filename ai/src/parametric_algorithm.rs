use crate::array;
use crate::linear_regression;

pub trait ParametricAlgorithm<T>: linear_regression::LinearRegressionModel<T> {
    fn theta(&self) -> array::ArrayView1<T>;
}
