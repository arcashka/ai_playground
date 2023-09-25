use crate::lms::kernel::*;

pub struct StochasticKernel;

impl<T> Kernel<T> for StochasticKernel
where
    T: num_traits::Float + num_traits::NumAssignOps + 'static,
{
    fn compute(
        x: ndarray::ArrayView2<T>,
        y: ndarray::ArrayView1<T>,
        settings: LMSSettingsFilled<T>,
    ) -> Result<LMSResult<T>, LMSError> {
        let m = x.nrows();
        let mut iteration_count = 0;
        let mut previous_cost = T::zero();
        let mut theta = settings.starting_theta.clone();
        loop {
            let mut cost = T::zero();
            for i in 0..m {
                let error = x.row(i).dot(&theta) - y[i];
                cost += error * error;
                theta.scaled_add((settings.learning_rate * error).neg(), &x.row(i));
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
}
