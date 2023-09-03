use crate::linear_regression;
use crate::training_data;

pub struct StochasticGradientDescent<T: nalgebra::RealField> {
    theta: nalgebra::DVector<T>,
    learning_rate: T,
    eps: T,
    max_iteration_count: usize,
}

impl<T: nalgebra::RealField + num::NumCast + num::Float> StochasticGradientDescent<T> {
    pub fn new(
        learning_rate: Option<T>,
        eps: Option<T>,
        max_iteration_count: Option<usize>,
    ) -> Option<Self> {
        let lr = num::cast::<f64, T>(0.001).unwrap();
        let epsilon = num::cast::<f64, T>(0.00001).unwrap();
        let zero = num::cast::<f64, T>(0.0).unwrap();
        Some(StochasticGradientDescent {
            theta: nalgebra::DVector::from_element(0, zero),
            learning_rate: learning_rate.unwrap_or(lr),
            eps: eps.unwrap_or(epsilon),
            max_iteration_count: max_iteration_count.unwrap_or(10000),
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
        let zero = num::cast::<f64, T>(0.0).unwrap();
        self.theta = match theta {
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
                let error = self.predict(&training_data.x.row(i)) - training_data.y[i];
                cost += error * error;
                for j in 0..n {
                    self.theta[j] -= self.learning_rate * error * training_data.x.row(i)[j];
                }
            }
            let cost_change = num::Float::abs(previous_cost - cost);
            let cost_change = cost_change / num::cast(m).unwrap();
            if cost_change < self.eps {
                break;
            }
            previous_cost = cost;
            if iteration_count >= self.max_iteration_count {
                break;
            }
            iteration_count += 1;
        }
        Ok(linear_regression::FittingInfo {
            theta: &self.theta,
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
        (x * &self.theta).x
    }
}
