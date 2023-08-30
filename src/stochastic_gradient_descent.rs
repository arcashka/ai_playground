use crate::linear_regression;
use crate::training_data;

pub struct StochasticGradientDescent {
    theta: Vec<f64>,
    learning_rate: f64,
    eps: f64,
    max_iteration_count: i32,
}

impl StochasticGradientDescent {
    pub fn new(
        learning_rate: Option<f64>,
        eps: Option<f64>,
        max_iteration_count: Option<i32>,
    ) -> Self {
        StochasticGradientDescent {
            theta: Vec::new(),
            learning_rate: learning_rate.unwrap_or(0.001),
            eps: eps.unwrap_or(0.00001),
            max_iteration_count: max_iteration_count.unwrap_or(10000),
        }
    }
}

impl linear_regression::LinearRegressionModel for StochasticGradientDescent {
    fn fit(
        &mut self,
        theta: Option<&[f64]>,
        training_data: &training_data::TrainingData,
    ) -> Result<linear_regression::FittingInfo, &'static str> {
        self.theta = match theta {
            Some(t) => {
                if t.len() != training_data.x_count {
                    return Err("Theta dimensions are not the same as training data");
                }
                t.to_vec()
            }
            None => vec![0.0; training_data.x_count],
        };
        let mut i = 0;
        let mut previous_cost = 0.0;
        loop {
            let mut cost = 0.0;
            for data in &training_data.examples {
                let error = self.predict(&data.x) - data.y;
                cost += error * error;
                self.theta
                    .iter_mut()
                    .zip(&data.x)
                    .for_each(|(theta_j, x_j)| {
                        *theta_j -= self.learning_rate * error * x_j;
                    });
            }
            let cost_change = (previous_cost - cost).abs() / training_data.examples.len() as f64;
            if cost_change < self.eps {
                break;
            }
            previous_cost = cost;
            if i >= self.max_iteration_count {
                break;
            }
            i += 1;
        }
        Ok(linear_regression::FittingInfo {
            theta: &self.theta,
            iteration_count: i,
        })
    }

    fn predict(&self, x: &[f64]) -> f64 {
        x.iter().zip(self.theta.iter()).map(|(&a, &b)| a * b).sum()
    }
}

