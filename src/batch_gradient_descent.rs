use crate::linear_regression;
use crate::training_data;

pub struct BatchGradientDescent {
    theta: Vec<f64>,
    learning_rate: f64,
    eps: f64,
    max_iteration_count: i32,
}

impl linear_regression::LinearRegressionModel for BatchGradientDescent {
    fn new(learning_rate: Option<f64>, eps: Option<f64>, max_iteration_count: Option<i32>) -> Self {
        BatchGradientDescent {
            theta: Vec::new(),
            learning_rate: learning_rate.unwrap_or(0.001),
            eps: eps.unwrap_or(0.00001),
            max_iteration_count: max_iteration_count.unwrap_or(10000),
        }
    }

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
        loop {
            let mut gradients: Vec<f64> = vec![0.0; training_data.x_count];
            for data in &training_data.examples {
                let error = self.predict(&data.x) - data.y;
                data.x
                    .iter()
                    .zip(gradients.iter_mut())
                    .for_each(|(&x, gradient)| {
                        *gradient += error * x;
                    })
            }
            self.theta
                .iter_mut()
                .zip(gradients.iter())
                .for_each(|(theta_j, gradient_j)| {
                    *theta_j -= self.learning_rate * gradient_j;
                });
            let common_error: f64 = gradients.iter().sum();
            if common_error.abs() < self.eps {
                break;
            }
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
