use crate::linear_regression::{LinearRegressionError, RealNumber};

pub struct BaseGradientDescent<T: RealNumber> {
    pub theta: Option<ndarray::Array1<T>>,
    pub learning_rate: T,
    pub eps: T,
    pub max_iteration_count: usize,
}

impl<T: RealNumber> BaseGradientDescent<T> {
    pub fn new(
        learning_rate: Option<T>,
        eps: Option<T>,
        max_iteration_count: Option<usize>,
    ) -> Result<Self, LinearRegressionError> {
        let lr = num::cast::<f64, T>(0.001).ok_or(LinearRegressionError::TypeError)?;
        let epsilon = num::cast::<f64, T>(0.00001).ok_or(LinearRegressionError::TypeError)?;
        Ok(BaseGradientDescent {
            theta: None,
            learning_rate: learning_rate.unwrap_or(lr),
            eps: eps.unwrap_or(epsilon),
            max_iteration_count: max_iteration_count.unwrap_or(10000),
        })
    }

    pub fn predict(&self, x: &ndarray::ArrayView1<T>) -> Result<T, LinearRegressionError> {
        let theta = self
            .theta
            .as_ref()
            .ok_or(LinearRegressionError::ThetaIsNotThereYet)?;
        Ok(x.dot(theta))
    }
}
