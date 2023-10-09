use crate::array;
use crate::linalg::arithmetic::Arithmetic;
use crate::linalg::dot::Dot;
use crate::lms::kernel::*;

pub struct BatchKernel;

impl<T> Kernel<T> for BatchKernel
where
    T: num_traits::Float + num_traits::NumAssignOps + std::iter::Sum,
{
    fn compute<F>(
        x: array::ArrayView2<T>,
        y: array::ArrayView1<T>,
        settings: LMSSettingsFilled<T>,
        weight_function: F,
    ) -> Result<LMSResult<T>, LMSError>
    where
        F: Fn(array::ArrayView1<T>) -> T,
    {
        let m = x.nrows();
        let n = x.ncols();
        let mut iteration_count = 0;
        let mut previous_cost = T::zero();
        let mut theta = settings.starting_theta.clone();
        loop {
            let mut gradients = array::Array1::<T>::zeros(n);
            let mut cost = T::zero();
            for i in 0..m {
                let weight = weight_function(x.row(i));
                let error = weight * x.row(i).dot(&theta) - y[i];
                cost += error * error;
                gradients = gradients.scaled_add(error, &x.row(i));
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
}
