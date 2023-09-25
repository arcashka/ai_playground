use ndarray_linalg::Inverse;

use crate::linear_regression;
use crate::linear_regression::LinearRegressionError;
use crate::parametric_algorithm;
use crate::training_data;

pub struct NormalEquation<T> {
    theta: ndarray::Array1<T>,
}

impl<T> linear_regression::LinearRegressionModel<T> for NormalEquation<T>
where
    T: num_traits::Float + 'static,
{
    fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        Ok(x.dot(&self.theta))
    }
}

impl<T> parametric_algorithm::ParametricAlgorithm<T> for NormalEquation<T>
where
    T: num_traits::Float + 'static,
{
    fn theta(&self) -> ndarray::ArrayView1<T> {
        self.theta.view()
    }
}

impl<T> NormalEquation<T>
where
    T: num_traits::Float + ndarray_linalg::Lapack,
{
    pub fn new(
        training_data: &training_data::TrainingData<T>,
    ) -> Result<Self, LinearRegressionError> {
        let x = training_data.x.view();
        let x_t = x.t();
        // θ = (XᵀX)⁻¹Xᵀy
        let theta = x_t
            .dot(&x)
            .inv()
            .map_err(|_| LinearRegressionError::OperationFailed)?
            .dot(&x_t.dot(&training_data.y));

        Ok(Self { theta })
    }
}
