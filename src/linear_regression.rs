use crate::training_data::TrainingData;

pub struct FittingInfo<'a, T> {
    pub theta: Option<ndarray::ArrayView1<'a, T>>,
    pub iteration_count: Option<usize>,
}

#[derive(Debug)]
pub enum LinearRegressionError {
    InvalidTheta,
    ProvidedThetaWillNotBeUsed,
    FailedToInvertMatrix,
    ThetaMissing,
    FailedCastToT,
    TrainingDataMissing,
}

pub trait LinearRegressionModel<T> {
    fn fit(
        &mut self,
        theta: Option<ndarray::Array1<T>>,
        training_data: &TrainingData<T>,
    ) -> Result<FittingInfo<T>, LinearRegressionError>;
    fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError>;
}
