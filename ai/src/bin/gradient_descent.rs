use ai::data_reader::cvs_reader;
use linalg::{algorithms::gradient_descent, ndarray::NDArray1};

fn main() {
    let training_data = cvs_reader::read_data::<f64>("resources/3.csv").unwrap();
    println!("X: {:?}", training_data.x);
    println!("Y: {:?}", training_data.y);

    let solver = gradient_descent::Builder::<f64, gradient_descent::BatchKernel>::default()
        .eps(0.001)
        .learning_rate(0.01)
        .max_iteration_count(1000)
        .build();
    solver.solve(
        training_data.x,
        training_data.y,
        None,
        Some(|_: &NDArray1<f64>| 1.0),
    );
}
