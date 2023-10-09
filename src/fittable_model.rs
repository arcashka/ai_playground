use crate::array;
use crate::{linear_regression, training_data};
use crate::{lms, parametric_algorithm};

#[derive(Debug)]
pub struct FittingInfo {
    pub iteration_count: usize,
}

pub struct FittingSettings<T> {
    pub max_iteration_count: usize,
    pub learning_rate: T,
    pub eps: T,
    pub starting_theta: array::Array1<T>,
}

pub trait FittableModel<T>: parametric_algorithm::ParametricAlgorithm<T> {
    fn fit<K>(
        training_data: &training_data::TrainingData<T>,
        settings: &FittingSettings<T>,
    ) -> Result<Self, linear_regression::LinearRegressionError>
    where
        Self: Sized,
        T: num_traits::Float + num_traits::NumAssignOps,
        K: lms::Kernel<T>;
    fn fitting_info(&self) -> FittingInfo;
}
