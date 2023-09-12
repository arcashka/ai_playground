mod base_gradient_descent;
mod batch_gradient_descent;
mod linear_regression;
//mod normal_equation;
mod stochastic_gradient_descent;
mod training_data;

use linear_regression::{LinearRegressionModel, RealNumber};
use std::error::Error;
use training_data::TrainingData;

fn run<T: RealNumber>(training_data: &TrainingData<T>, model: &mut dyn LinearRegressionModel<T>) {
    let fitting_info = model.fit(None, &training_data);
    match fitting_info {
        Ok(result) => println!(
            "Iterations: {}\ntheta: {:?}",
            result.iteration_count, result.theta
        ),
        Err(error) => println!("Error: {}", error),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let training_data = training_data::read_data::<f64>("resources/3.csv")?;
    let mut batch_gradient_descent = batch_gradient_descent::BatchGradientDescent::<f64>::new(
        Some(0.001),
        Some(0.00001),
        Some(10000),
    )?;
    run(&training_data, &mut batch_gradient_descent);
    let mut stochastic_gradient_descent = stochastic_gradient_descent::StochasticGradientDescent::<
        f64,
    >::new(Some(0.001), Some(0.000001), Some(10000))?;
    run(&training_data, &mut stochastic_gradient_descent);
    Ok(())
}
