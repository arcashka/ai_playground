use super::{BatchKernel, Builder, Kernel, Solver, StochasticKernel};

impl Kernel for BatchKernel {}
impl Kernel for StochasticKernel {}

impl<A, K: Kernel> Default for Builder<A, K>
where
    K: Kernel,
{
    fn default() -> Self {
        Self {
            kernel: std::marker::PhantomData,
            max_iteration_count: None,
            learning_rate: None,
            eps: None,
        }
    }
}

impl<A, K: Kernel> Builder<A, K> {
    pub fn max_iteration_count(mut self, i: usize) -> Self {
        self.max_iteration_count = Some(i);
        self
    }

    pub fn learning_rate(mut self, learning_rate: A) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn eps(mut self, eps: A) -> Self {
        self.eps = Some(eps);
        self
    }

    pub fn build(self) -> Solver<A, K>
    where
        A: From<f64>,
    {
        Solver::<A, K> {
            kernel: std::marker::PhantomData,
            max_iteration_count: self.max_iteration_count.unwrap_or(10000),
            eps: self.eps.unwrap_or(0.001.into()),
            learning_rate: self.learning_rate.unwrap_or(0.01.into()),
        }
    }
}
