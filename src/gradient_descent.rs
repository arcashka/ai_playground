use crate::array;
use crate::fittable_model;
use crate::linalg::dot::Dot;
use crate::linear_regression;
use crate::linear_regression::LinearRegressionError;
use crate::lms;
use crate::parametric_algorithm;
use crate::training_data;

pub struct GradientDescent<T> {
    theta: array::Array1<T>,
    iteration_count: usize,
}

impl From<lms::LMSError> for LinearRegressionError {
    fn from(error: lms::LMSError) -> Self {
        match error {
            lms::LMSError::FailedCastToT => LinearRegressionError::FailedCastToT,
        }
    }
}

impl<T> linear_regression::LinearRegressionModel<T> for GradientDescent<T>
where
    T: num_traits::Float + std::iter::Sum,
{
    fn predict(&self, x: &array::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        Ok(x.dot(&self.theta))
    }
}

impl<T> parametric_algorithm::ParametricAlgorithm<T> for GradientDescent<T>
where
    T: num_traits::Float + std::iter::Sum,
{
    fn theta(&self) -> array::ArrayView1<T> {
        self.theta.view()
    }
}

impl<T> fittable_model::FittableModel<T> for GradientDescent<T>
where
    T: num_traits::Float + num_traits::NumAssignOps + std::iter::Sum,
{
    fn fit<K>(
        training_data: &training_data::TrainingData<T>,
        settings: &fittable_model::FittingSettings<T>,
    ) -> Result<Self, LinearRegressionError>
    where
        K: lms::Kernel<T>,
    {
        let settings = lms::LMSSettings {
            max_iteration_count: Some(settings.max_iteration_count),
            learning_rate: Some(settings.learning_rate),
            eps: Some(settings.eps),
            starting_theta: Some(settings.starting_theta.clone()),
        };
        let lms_result = lms::lms_solve::<T, K, _>(
            training_data.x.view(),
            training_data.y.view(),
            Some(settings),
            |_| T::one(),
        )?;
        Ok(Self {
            theta: lms_result.theta,
            iteration_count: lms_result.iteration_count,
        })
    }

    fn fitting_info(&self) -> fittable_model::FittingInfo {
        fittable_model::FittingInfo {
            iteration_count: self.iteration_count,
        }
    }
}
