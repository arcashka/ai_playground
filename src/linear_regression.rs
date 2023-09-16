use crate::training_data::TrainingData;

pub trait RealNumber: nalgebra::RealField + num::Float + std::str::FromStr {}
impl<T: nalgebra::RealField + num::Float + std::str::FromStr> RealNumber for T {}

pub struct FittingInfo<'a, T: RealNumber> {
    pub theta: ndarray::ArrayView1<'a, T>,
    pub iteration_count: usize,
}

#[derive(Debug)]
pub enum LinearRegressionError {
    InvalidTheta,
    ThetaIsNotThereYet,
    TypeError,
}

pub trait LinearRegressionModel<T: RealNumber> {
    fn fit(
        &mut self,
        theta: Option<ndarray::Array1<T>>,
        training_data: &TrainingData<T>,
    ) -> Result<FittingInfo<T>, LinearRegressionError>;
    fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError>;
}
