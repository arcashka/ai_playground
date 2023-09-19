use crate::training_data::TrainingData;

pub trait Float: ndarray::NdFloat + std::str::FromStr {}
impl<T: ndarray::NdFloat + std::str::FromStr> Float for T {}

pub struct FittingInfo<'a, T: Float> {
    pub theta: ndarray::ArrayView1<'a, T>,
    pub iteration_count: usize,
}

#[derive(Debug)]
pub enum LinearRegressionError {
    InvalidTheta,
    ThetaIsNotThereYet,
    ProvidedThetaWillNotBeUsed,
    TypeError,
}

pub trait LinearRegressionModel<T: Float> {
    fn fit(
        &mut self,
        theta: Option<ndarray::Array1<T>>,
        training_data: &TrainingData<T>,
    ) -> Result<FittingInfo<T>, LinearRegressionError>;
    fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError>;
}
