use crate::array;
use crate::fittable_model;
use crate::linalg::Arithmetic;
use crate::linalg::Dot;
use crate::linear_regression;
use crate::linear_regression::LinearRegressionError;
use crate::lms;
use crate::training_data;

pub struct LocallyWeightedLinearRegression<T> {
    training_data: training_data::TrainingData<T>,
    settings: Settings<T>,
}

pub struct Settings<T> {
    pub common_settings: fittable_model::FittingSettings<T>,
    pub bandwith: T,
}

impl<T> LocallyWeightedLinearRegression<T>
where
    T: Clone,
{
    pub fn new(data: &training_data::TrainingData<T>, settings: Settings<T>) -> Self {
        Self {
            training_data: data.to_owned(),
            settings,
        }
    }
}

impl<T> linear_regression::LinearRegressionModel<T> for LocallyWeightedLinearRegression<T>
where
    T: num_traits::Float + num_traits::NumAssignOps + num_traits::FloatConst + std::iter::Sum,
{
    fn predict(&self, x_i: &array::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        let settings = lms::LMSSettings {
            max_iteration_count: Some(self.settings.common_settings.max_iteration_count),
            learning_rate: Some(self.settings.common_settings.learning_rate),
            eps: Some(self.settings.common_settings.eps),
            starting_theta: Some(self.settings.common_settings.starting_theta.to_owned()),
        };
        let lms_result = lms::lms_solve::<T, lms::BatchKernel, _>(
            self.training_data.x.view(),
            self.training_data.y.view(),
            Some(settings),
            |x| {
                // L2||x_i - x||
                let diff = x_i.sub(&x);
                let distance: T = diff.dot(&diff);
                let two = T::one() + T::one();
                // e^{-\frac{distance^2}{2*\tau^2}}
                T::E().powf(distance / two * self.settings.bandwith.powi(2).neg())
            },
        )?;
        let result = x_i.dot(&lms_result.theta);
        Ok(result)
    }
}
