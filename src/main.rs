mod batch_gradient_descent;
mod linear_regression;
mod stochastic_gradient_descent;
mod training_data;

use linear_regression::LinearRegressionModel;
use training_data::TrainingData;

fn run(training_data: &TrainingData, model: &mut dyn LinearRegressionModel) {
    let fitting_info = model.fit(None, &training_data);
    match fitting_info {
        Ok(result) => println!(
            "Iterations: {}\ntheta: {:?}",
            result.iteration_count, result.theta
        ),
        Err(error) => println!("Error: {}", error),
    }
}

fn main() {
    let training_data = training_data::read_data("resources/3.csv").unwrap();
    let mut batch_gradient_descent =
        batch_gradient_descent::BatchGradientDescent::new(Some(0.001), Some(0.000001), Some(10000));
    run(&training_data, &mut batch_gradient_descent);
    let mut stochastic_gradient_descent =
        stochastic_gradient_descent::StochasticGradientDescent::new(
            Some(0.001),
            Some(0.000001),
            Some(10000),
        );
    run(&training_data, &mut stochastic_gradient_descent);
}
