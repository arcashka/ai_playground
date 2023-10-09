mod fittable_model;
mod gradient_descent;
mod linear_regression;
mod lms;
mod locally_weighted_gradient_descent;
mod normal_equation;
mod parametric_algorithm;
mod training_data;

mod array;

mod linalg;

mod window;

use fittable_model::FittableModel;
use linear_regression::{LinearRegressionError, LinearRegressionModel};
use parametric_algorithm::ParametricAlgorithm;
use training_data::TrainingDataError;

use std::convert::From;

#[derive(Debug)]
pub enum MainError {
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

fn print<T>(
    name: &str,
    theta: array::ArrayView1<T>,
    fitting_info: Option<fittable_model::FittingInfo>,
) where
    T: num_traits::Float + std::fmt::Debug,
{
    println!("{}", name);
    match fitting_info {
        Some(info) => {
            println!("{:?}", info);
        }
        None => (),
    };
    println!("{:?}\n", theta);
}

pub fn run() -> Result<(), MainError> {
    let training_data = training_data::read_data::<f64>("resources/3.csv")?;
    let fitting_settings = fittable_model::FittingSettings {
        max_iteration_count: 10000,
        learning_rate: 0.001,
        eps: 0.00001,
        starting_theta: array::Array1::<f64>::zeros(training_data.x.ncols()),
    };
    let batch_gradient_descent = gradient_descent::GradientDescent::<f64>::fit::<lms::BatchKernel>(
        &training_data,
        &fitting_settings,
    )?;
    print(
        "batch gradient descent",
        batch_gradient_descent.theta(),
        Some(batch_gradient_descent.fitting_info()),
    );
    let stochastic_gradient_descent = gradient_descent::GradientDescent::<f64>::fit::<
        lms::StochasticKernel,
    >(&training_data, &fitting_settings)?;
    print(
        "stochastic gradient descent",
        stochastic_gradient_descent.theta(),
        Some(stochastic_gradient_descent.fitting_info()),
    );

    let normal_equation_solver = normal_equation::NormalEquation::<f64>::new(&training_data)?;
    print("normal equations", normal_equation_solver.theta(), None);

    let locally_weighted_gradient_descent =
        locally_weighted_gradient_descent::LocallyWeightedLinearRegression::<f64>::new(
            &training_data,
            locally_weighted_gradient_descent::Settings {
                common_settings: fitting_settings,
                bandwith: 1.0,
            },
        );
    println!(
        "locally weighted gradient descent\n{:?}\n",
        locally_weighted_gradient_descent.predict(&training_data.x.row(0))?,
    );
    window::run();
    Ok(())
}
