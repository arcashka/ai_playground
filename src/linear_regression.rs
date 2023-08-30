use crate::training_data::TrainingData;

pub struct FittingInfo<'a> {
    pub theta: &'a [f64],
    pub iteration_count: i32,
}

pub trait LinearRegressionModel {
    fn fit(
        &mut self,
        theta: Option<&[f64]>,
        training_data: &TrainingData,
    ) -> Result<FittingInfo, &'static str>;
    fn predict(&self, x: &[f64]) -> f64;
}
