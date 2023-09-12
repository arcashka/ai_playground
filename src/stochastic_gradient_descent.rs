use crate::base_gradient_descent;
use crate::linear_regression;
use crate::training_data;

pub struct StochasticGradientDescent<T: nalgebra::RealField> {
    base: base_gradient_descent::BaseGradientDescent<T>,
}

impl<T: nalgebra::RealField + num::NumCast + num::Float> StochasticGradientDescent<T> {
    pub fn new(
        learning_rate: Option<T>,
        eps: Option<T>,
        max_iteration_count: Option<usize>,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            base: base_gradient_descent::BaseGradientDescent::new(
                learning_rate,
                eps,
                max_iteration_count,
            )?,
        })
    }
}

impl<T> linear_regression::LinearRegressionModel<T> for StochasticGradientDescent<T>
where
    T: nalgebra::RealField + std::iter::Sum + num::Float + Copy,
{
    fn fit(
        &mut self,
        theta: Option<nalgebra::DVector<T>>,
        training_data: &training_data::TrainingData<T>,
    ) -> Result<linear_regression::FittingInfo<T>, &'static str> {
        let n = training_data.x.ncols();
        let m = training_data.x.nrows();
        let zero = num::cast::<f64, T>(0.0).ok_or("failed to cast to T")?;
        self.base.theta = match theta {
            Some(t) => {
                if t.len() != n {
                    return Err("Theta dimensions are not the same as training data");
                }
                t
            }
            None => nalgebra::DVector::<T>::from_element(n, zero),
        };
        let mut iteration_count = 0;
        let mut previous_cost = zero;
        loop {
            let mut cost = zero;
            for i in 0..m {
                let error = self.base.predict(&training_data.x.row(i)) - training_data.y[i];
                cost += error * error;
                for j in 0..n {
                    self.base.theta[j] -=
                        self.base.learning_rate * error * training_data.x.row(i)[j];
                }
            }
            let cost_change = num::Float::abs(previous_cost - cost);
            let cost_change = cost_change / num::cast(m).ok_or("Failed to cast to T")?;
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
            theta: &self.base.theta,
            iteration_count,
        })
    }

    fn predict(
        &self,
        x: &nalgebra::RowVector<
            T,
            nalgebra::Dyn,
            nalgebra::ViewStorage<T, nalgebra::U1, nalgebra::Dyn, nalgebra::U1, nalgebra::Dyn>,
        >,
    ) -> T {
        self.base.predict(x)
    }
}
