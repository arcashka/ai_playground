use crate::linear_regression::RealNumber;

pub struct BaseGradientDescent<T: RealNumber> {
    pub theta: nalgebra::DVector<T>,
    pub learning_rate: T,
    pub eps: T,
    pub max_iteration_count: usize,
}

impl<T: RealNumber> BaseGradientDescent<T> {
    pub fn new(
        learning_rate: Option<T>,
        eps: Option<T>,
        max_iteration_count: Option<usize>,
    ) -> Result<Self, &'static str> {
        let lr = num::cast::<f64, T>(0.001).ok_or("failed to cast learning rate to T")?;
        let epsilon = num::cast::<f64, T>(0.00001).ok_or("failed to cast eps to T")?;
        let zero = num::cast::<f64, T>(0.0).ok_or("failed to cast zero to T")?;
        Ok(BaseGradientDescent {
            theta: nalgebra::DVector::from_element(0, zero),
            learning_rate: learning_rate.unwrap_or(lr),
            eps: eps.unwrap_or(epsilon),
            max_iteration_count: max_iteration_count.unwrap_or(10000),
        })
    }

    pub fn predict(
        &self,
        x: &nalgebra::RowVector<
            T,
            nalgebra::Dyn,
            nalgebra::ViewStorage<T, nalgebra::U1, nalgebra::Dyn, nalgebra::U1, nalgebra::Dyn>,
        >,
    ) -> T {
        (x * &self.theta).x
    }
}
