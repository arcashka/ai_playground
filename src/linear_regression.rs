use crate::training_data::TrainingData;

pub trait RealNumber: nalgebra::RealField + num::Float + std::str::FromStr {}
impl<T: nalgebra::RealField + num::Float + std::str::FromStr> RealNumber for T {}

pub struct FittingInfo<'a, T: RealNumber> {
    pub theta: &'a nalgebra::DVector<T>,
    pub iteration_count: usize,
}

pub trait LinearRegressionModel<T: RealNumber> {
    fn fit(
        &mut self,
        theta: Option<nalgebra::DVector<T>>,
        training_data: &TrainingData<T>,
    ) -> Result<FittingInfo<T>, &'static str>;
    fn predict(
        &self,
        vec: &nalgebra::RowVector<
            T,
            nalgebra::Dyn,
            nalgebra::ViewStorage<T, nalgebra::U1, nalgebra::Dyn, nalgebra::U1, nalgebra::Dyn>,
        >,
    ) -> T;
}
