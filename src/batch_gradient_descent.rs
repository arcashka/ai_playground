use crate::linear_regression;
use crate::linear_regression::LinearRegressionError;
use crate::lms;
use crate::parametric_gradient_descent;
use crate::training_data;

pub struct BatchGradientDescent<T> {
    base: parametric_gradient_descent::ParametricGradientDescent<T>,
}

impl<T: num_traits::Float> BatchGradientDescent<T> {
    pub fn new(
        learning_rate: Option<T>,
        eps: Option<T>,
        max_iteration_count: Option<usize>,
    ) -> Result<Self, LinearRegressionError> {
        Ok(Self {
            base: parametric_gradient_descent::ParametricGradientDescent::new(
                learning_rate,
                eps,
                max_iteration_count,
            )?,
        })
    }
}

impl From<lms::LMSError> for LinearRegressionError {
    fn from(error: lms::LMSError) -> Self {
        match error {
            lms::LMSError::InvalidTheta => LinearRegressionError::InvalidTheta,
            lms::LMSError::FailedCastToT => LinearRegressionError::FailedCastToT,
        }
    }
}

impl<T> linear_regression::LinearRegressionModel<T> for BatchGradientDescent<T>
where
    T: num_traits::Float + num_traits::NumAssignOps + 'static,
{
    fn fit(
        &mut self,
        theta: Option<ndarray::Array1<T>>,
        training_data: &training_data::TrainingData<T>,
    ) -> Result<linear_regression::FittingInfo<T>, LinearRegressionError> {
        let settings = lms::LMSSettings {
            max_iteration_count: Some(self.base.max_iteration_count),
            learning_rate: Some(self.base.learning_rate),
            eps: Some(self.base.eps),
            starting_theta: theta,
        };
        let lms_result = lms::lms_solve(
            training_data.x.view(),
            training_data.y.view(),
            Some(settings),
        )?;
        self.base.theta = Some(lms_result.theta);
        // TODO: result fields shouldnt be Option
        Ok(linear_regression::FittingInfo {
            theta: Some(self.base.theta.as_ref().expect("just put it there").view()),
            iteration_count: Some(lms_result.iteration_count),
        })
    }

    fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        self.base.predict(x)
    }
}
