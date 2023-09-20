use ndarray_linalg::Inverse;

use crate::linear_regression::{self, LinearRegressionError};
use crate::training_data;

pub struct NormalEquation<T: linear_regression::Float> {
    pub theta: Option<ndarray::Array1<T>>,
}

impl<T: linear_regression::Float> NormalEquation<T> {
    pub fn new() -> Self {
        Self { theta: None }
    }
}

impl<T> linear_regression::LinearRegressionModel<T> for NormalEquation<T>
where
    T: linear_regression::Float,
{
    fn fit(
        &mut self,
        theta: Option<ndarray::Array1<T>>,
        training_data: &training_data::TrainingData<T>,
    ) -> Result<linear_regression::FittingInfo<T>, LinearRegressionError> {
        if theta.is_some() {
            return Err(LinearRegressionError::ProvidedThetaWillNotBeUsed);
        };
        let x = training_data.x.view();
        let x_t = x.t();
        self.theta = Some(
            x_t.dot(&x)
                .inv()
                .map_err(|_| LinearRegressionError::FailedToInvertMatrix)?
                .dot(&x_t.dot(&training_data.y)),
        );

        Ok(linear_regression::FittingInfo {
            theta: self
                .theta
                .as_ref()
                .ok_or(LinearRegressionError::ThetaIsNotThereYet)?
                .view(),
            iteration_count: 1,
        })
    }

    fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        let theta = self
            .theta
            .as_ref()
            .ok_or(LinearRegressionError::ThetaIsNotThereYet)?;
        Ok(x.dot(theta))
    }
}
