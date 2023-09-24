use cauchy::Scalar;
use ndarray_linalg::Norm;
use std::ops::Sub;

use crate::linear_regression;
use crate::linear_regression::LinearRegressionError;
use crate::training_data;

pub struct LocallyWeightedLinearRegression<T> {
    training_data: Option<training_data::TrainingData<T>>,
    bandwith: T,
}

impl<T> LocallyWeightedLinearRegression<T>
where
    T: num_traits::Float + num_traits::FloatConst + ndarray_linalg::Lapack,
{
    pub fn new() -> Self {
        Self {
            training_data: None,
            bandwith: T::one(),
        }
    }

    fn training_data_ref(&self) -> Result<&training_data::TrainingData<T>, LinearRegressionError> {
        self.training_data
            .as_ref()
            .ok_or(LinearRegressionError::TrainingDataMissing)
    }

    fn weight(&self, x: ndarray::ArrayView1<T>, x_i: ndarray::ArrayView1<T>) -> T {
        // L2||x_i - x||
        let distance: T = Scalar::powi(T::from_real(x_i.sub(&x).norm()), 2);
        let two = T::one() + T::one();
        // e^{-\frac{distance^2}{2*\tau^2}}
        num::Float::powf(
            T::E(),
            (distance / two * Scalar::powi(self.bandwith, 2)).neg(),
        )
    }
}

impl<T> linear_regression::LinearRegressionModel<T> for LocallyWeightedLinearRegression<T>
where
    T: num_traits::Float,
{
    fn fit(
        &mut self,
        theta: Option<ndarray::Array1<T>>,
        training_data: &training_data::TrainingData<T>,
    ) -> Result<linear_regression::FittingInfo<T>, LinearRegressionError> {
        if theta.is_some() {
            return Err(LinearRegressionError::ProvidedThetaWillNotBeUsed);
        };
        self.training_data = Some(training_data.to_owned());
        Ok(linear_regression::FittingInfo {
            theta: None,
            iteration_count: None,
        })
    }

    fn predict(&self, _: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        Ok(T::zero())
    }
}
