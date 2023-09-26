mod batch_kernel;
mod kernel;
mod stochastic_kernel;

use crate::lms::kernel::LMSSettingsFilled;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use num_traits;

pub use crate::lms::batch_kernel::BatchKernel;
pub use crate::lms::kernel::Kernel;
pub use crate::lms::stochastic_kernel::StochasticKernel;

pub use crate::lms::kernel::LMSError;
pub use crate::lms::kernel::LMSResult;

pub struct LMSSettings<T> {
    pub max_iteration_count: Option<usize>,
    pub learning_rate: Option<T>,
    pub eps: Option<T>,
    pub starting_theta: Option<Array1<T>>,
}

fn fill_missing_settings<T>(
    settings: Option<LMSSettings<T>>,
    theta_dimensions: usize,
) -> Result<LMSSettingsFilled<T>, LMSError>
where
    T: num::Float,
{
    let settings = match settings {
        Some(t) => t,
        None => LMSSettings::<T> {
            max_iteration_count: Some(10000),
            learning_rate: Some(T::from(0.001).ok_or(LMSError::FailedCastToT)?),
            eps: Some(T::from(0.00001).ok_or(LMSError::FailedCastToT)?),
            starting_theta: None,
        },
    };
    Ok(LMSSettingsFilled::<T> {
        max_iteration_count: settings.max_iteration_count.unwrap_or(10000),
        learning_rate: settings
            .learning_rate
            .unwrap_or(T::from(0.001).ok_or(LMSError::FailedCastToT)?),
        eps: settings
            .learning_rate
            .unwrap_or(T::from(0.00001).ok_or(LMSError::FailedCastToT)?),
        starting_theta: settings
            .starting_theta
            .unwrap_or(Array1::zeros(theta_dimensions)),
    })
}

pub fn lms_solve<'a, T, K, F>(
    x: ArrayView2<'a, T>,
    y: ArrayView1<'a, T>,
    settings: Option<LMSSettings<T>>,
    weight_function: F,
) -> Result<LMSResult<T>, LMSError>
where
    T: num_traits::Float + num_traits::NumAssignOps + 'static,
    K: kernel::Kernel<T>,
    F: Fn(ndarray::ArrayView1<T>) -> T,
{
    let settings = fill_missing_settings(settings, x.ncols())?;
    K::compute(x, y, settings, weight_function)
}
