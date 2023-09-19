mod base_gradient_descent;
mod batch_gradient_descent;
mod linear_regression;
//mod normal_equation;
mod stochastic_gradient_descent;
mod training_data;

use linear_regression::{LinearRegressionError, LinearRegressionModel};
use training_data::{TrainingData, TrainingDataError};

use std::convert::From;

#[derive(Debug)]
enum MainError {
    LinearRegressionError(LinearRegressionError),
    TrainingDataError(TrainingDataError),
}

impl From<LinearRegressionError> for MainError {
    fn from(error: LinearRegressionError) -> Self {
        MainError::LinearRegressionError(error)
    }
}

impl From<TrainingDataError> for MainError {
    fn from(error: TrainingDataError) -> Self {
        MainError::TrainingDataError(error)
    }
}

fn run<T: linear_regression::Float>(
    training_data: &TrainingData<T>,
    model: &mut dyn LinearRegressionModel<T>,
) -> Result<(), LinearRegressionError> {
    let fitting_info = model.fit(None, &training_data)?;
    println!(
        "Iterations: {}\ntheta: {:?}",
        fitting_info.iteration_count, fitting_info.theta
    );
    Ok(())
}

fn main() -> Result<(), MainError> {
    let training_data = training_data::read_data::<f64>("resources/3.csv")?;
    let mut batch_gradient_descent = batch_gradient_descent::BatchGradientDescent::<f64>::new(
        Some(0.001),
        Some(0.00001),
        Some(10000),
    )?;
    run(&training_data, &mut batch_gradient_descent)?;
    let mut stochastic_gradient_descent = stochastic_gradient_descent::StochasticGradientDescent::<
        f64,
    >::new(Some(0.001), Some(0.000001), Some(10000))?;
    run(&training_data, &mut stochastic_gradient_descent)?;
    Ok(())
}
