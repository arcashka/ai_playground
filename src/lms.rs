use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use num_traits;

#[derive(Debug)]
pub enum LMSError {
    InvalidTheta,
    FailedCastToT,
}

pub struct LMSSettings<T> {
    pub max_iteration_count: Option<usize>,
    pub learning_rate: Option<T>,
    pub eps: Option<T>,
    pub starting_theta: Option<Array1<T>>,
}

struct LMSSettingsFilled<T> {
    max_iteration_count: usize,
    learning_rate: T,
    eps: T,
    starting_theta: Array1<T>,
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

pub struct LMSResult<T> {
    pub theta: Array1<T>,
    pub iteration_count: usize,
}

pub fn lms_solve<'a, T>(
    x: ArrayView2<'a, T>,
    y: ArrayView1<'a, T>,
    settings: Option<LMSSettings<T>>,
) -> Result<LMSResult<T>, LMSError>
where
    T: num_traits::Float + num_traits::NumAssignOps + 'static,
{
    let m = x.nrows();
    let n = x.ncols();

    let settings = fill_missing_settings(settings, n)?;
    let mut theta = settings.starting_theta;

    let mut iteration_count = 0;
    let mut previous_cost = T::zero();
    loop {
        let mut gradients = ndarray::Array1::<T>::zeros(n);
        let mut cost = T::zero();
        for i in 0..m {
            let error = x.row(i).dot(&theta) - y[i];
            cost += error * error;
            gradients.scaled_add(error, &x.row(i));
        }
        for i in 0..n {
            theta[i] -= settings.learning_rate * gradients[i];
        }
        let cost_change = num::Float::abs(previous_cost - cost);
        let cost_change = cost_change / T::from(m).ok_or(LMSError::FailedCastToT)?;
        if cost_change < settings.eps {
            break;
        }
        previous_cost = cost;
        if iteration_count >= settings.max_iteration_count {
            break;
        }
        iteration_count += 1;
    }

    Ok(LMSResult {
        theta,
        iteration_count,
    })
}
