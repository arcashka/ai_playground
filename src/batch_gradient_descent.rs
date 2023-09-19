use crate::base_gradient_descent;
use crate::linear_regression;
use crate::linear_regression::LinearRegressionError;
use crate::training_data;

pub struct BatchGradientDescent<T: linear_regression::Float> {
    base: base_gradient_descent::BaseGradientDescent<T>,
}

impl<T: linear_regression::Float> BatchGradientDescent<T> {
    pub fn new(
        learning_rate: Option<T>,
        eps: Option<T>,
        max_iteration_count: Option<usize>,
    ) -> Result<Self, LinearRegressionError> {
        Ok(Self {
            base: base_gradient_descent::BaseGradientDescent::new(
                learning_rate,
                eps,
                max_iteration_count,
            )?,
        })
    }
}

impl<T> linear_regression::LinearRegressionModel<T> for BatchGradientDescent<T>
where
    T: linear_regression::Float,
{
    fn fit(
        &mut self,
        theta: Option<ndarray::Array1<T>>,
        training_data: &training_data::TrainingData<T>,
    ) -> Result<linear_regression::FittingInfo<T>, LinearRegressionError> {
        let m = training_data.x.nrows();
        let n = training_data.x.ncols();
        let zero = num::cast::<f64, T>(0.0).ok_or(LinearRegressionError::TypeError)?;
        self.base.theta = match theta {
            Some(t) => {
                if t.len() != n {
                    return Err(LinearRegressionError::InvalidTheta);
                }
                Some(t)
            }
            None => Some(ndarray::Array1::<T>::zeros(n)),
        };
        let theta_ref = self
            .base
            .theta
            .as_mut()
            .ok_or(LinearRegressionError::InvalidTheta)?;
        let mut iteration_count = 0;
        let mut previous_cost = zero;
        loop {
            let mut gradients: Vec<T> = vec![zero; n];
            let mut cost = zero;
            for i in 0..m {
                let error = training_data.x.row(i).dot(theta_ref) - training_data.y[i];
                cost += error * error;
                for j in 0..n {
                    gradients[j] += error * training_data.x.row(i)[j];
                }
            }
            for i in 0..n {
                theta_ref[i] -= self.base.learning_rate * gradients[i];
            }
            let cost_change: T = num::Float::abs(previous_cost - cost);
            let cost_change = cost_change / num::cast(m).ok_or(LinearRegressionError::TypeError)?;
            if cost_change < self.base.eps {
                break;
            }
            previous_cost = cost;
            if iteration_count >= self.base.max_iteration_count {
                break;
            }
            iteration_count += 1;
        }
        Ok(linear_regression::FittingInfo {
            theta: theta_ref.view(),
            iteration_count,
        })
    }

    fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        self.base.predict(x)
    }
}
