use crate::array;
use crate::array::Transpose;
use crate::linalg::Dot;
use crate::linalg::LLT;
use crate::linear_regression;
use crate::linear_regression::LinearRegressionError;
use crate::parametric_algorithm;
use crate::training_data;

pub struct NormalEquation<T> {
    theta: array::Array1<T>,
}

impl<T> linear_regression::LinearRegressionModel<T> for NormalEquation<T>
where
    T: num_traits::Float + num_traits::NumAssign,
{
    fn predict(&self, x: &array::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        Ok(x.dot(&self.theta))
    }
}

impl<T> parametric_algorithm::ParametricAlgorithm<T> for NormalEquation<T>
where
    T: num_traits::Float + num_traits::NumAssign,
{
    fn theta(&self) -> array::ArrayView1<T> {
        self.theta.view()
    }
}

impl<T> NormalEquation<T>
where
    T: num_traits::Float + std::iter::Sum,
{
    pub fn new(
        training_data: &training_data::TrainingData<T>,
    ) -> Result<Self, LinearRegressionError> {
        let x = training_data.x.view();
        let x_t = x.t();
        let y_as_matrix = training_data
            .y
            .clone()
            .into_shape((training_data.y.len(), 1))?;
        // θ = (XᵀX)⁻¹Xᵀy
        let theta = x_t.dot(&x).llt().inv().dot(&x_t).dot(&y_as_matrix);

        let theta_len = theta.len();
        Ok(Self {
            theta: theta.into_shape(theta_len)?,
        })
    }
}

impl From<array::ShapeError> for LinearRegressionError {
    fn from(_: array::ShapeError) -> Self {
        LinearRegressionError::OperationFailed
    }
}
