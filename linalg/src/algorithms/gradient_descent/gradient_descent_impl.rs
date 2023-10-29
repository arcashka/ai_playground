use super::{Error, Kernel, Solver};
use crate::ndarray::{NDArray1, NDArray2};

impl<A, K: Kernel> Solver<A, K> {
    pub fn solve<F>(
        &self,
        x: NDArray2<A>,
        y: NDArray1<A>,
        starting_theta: Option<NDArray1<A>>,
        weight_function: Option<F>,
    ) -> Result<NDArray1<A>, Error>
    where
        F: Fn(&NDArray1<A>) -> A,
        A: num_traits::Zero + num_traits::One,
    {
        let m = x.nrows();
        let n = x.ncols();

        let mut iteration_count = 0;
        let mut previous_cost = A::zero();
        let mut theta = starting_theta.unwrap_or(NDArray1::<A>::zeros([n]));

        loop {
            let mut gradients = NDArray1::<A>::zeros([n]);
            let mut cost = A::zero();
            for i in 0..m {
                let weight = if let Some(weight_func) = &weight_function {
                    weight_func(&x.row(i))
                } else {
                    A::one() // Assuming A also implements num_traits::One
                };
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
