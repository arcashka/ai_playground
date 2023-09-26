pub struct LMSSettingsFilled<T> {
    pub max_iteration_count: usize,
    pub learning_rate: T,
    pub eps: T,
    pub starting_theta: ndarray::Array1<T>,
}

pub struct LMSResult<T> {
    pub theta: ndarray::Array1<T>,
    pub iteration_count: usize,
}

#[derive(Debug)]
pub enum LMSError {
    FailedCastToT,
}

pub trait Kernel<T> {
    fn compute<F>(
        x: ndarray::ArrayView2<T>,
        y: ndarray::ArrayView1<T>,
        settings: LMSSettingsFilled<T>,
        weight_function: F,
    ) -> Result<LMSResult<T>, LMSError>
    where
        T: num_traits::Float + num_traits::NumAssignOps,
        F: Fn(ndarray::ArrayView1<T>) -> T;
}
