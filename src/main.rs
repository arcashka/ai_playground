mod batch_gradient_descent;
mod linear_regression;
mod training_data;

use linear_regression::LinearRegressionModel;

fn main() {
    let training_data = training_data::read_data("resources/3.csv").unwrap();
    let mut regression =
        batch_gradient_descent::BatchGradientDescent::new(Some(0.001), Some(0.000001), Some(10000));
    let fitting_info = regression.fit(None, &training_data);
    match fitting_info {
        Ok(result) => println!(
            "Iterations: {}\ntheta: {:?}",
            result.iteration_count, result.theta
        ),
        Err(error) => println!("Error: {}", error),
    }
}
