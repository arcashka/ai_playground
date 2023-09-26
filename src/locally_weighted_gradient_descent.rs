use cauchy::Scalar;
use ndarray_linalg::Lapack;
use ndarray_linalg::Norm;
use std::ops::Sub;

use crate::fittable_model;
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
    T: num_traits::Float + num_traits::NumAssignOps + Lapack + num_traits::FloatConst,
{
    fn predict(&self, x_i: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError> {
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
                let distance: T = Scalar::powi(T::from_real(x_i.sub(&x).norm()), 2);
                let two = T::one() + T::one();
                // e^{-\frac{distance^2}{2*\tau^2}}
                num::Float::powf(
                    T::E(),
                    (distance / two * Scalar::powi(self.settings.bandwith, 2)).neg(),
                )
            },
        )?;
        let result = x_i.dot(&lms_result.theta);
        Ok(result)
    }
}
