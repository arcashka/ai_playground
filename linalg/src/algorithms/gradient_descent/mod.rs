use crate::ndarray::NDArray1;

mod builder_impl;
mod gradient_descent_impl;

pub enum Error {}

pub trait Kernel {}
pub struct BatchKernel;
pub struct StochasticKernel;

pub struct Solver<A, K>
where
    K: Kernel,
{
    kernel: std::marker::PhantomData<K>,
    max_iteration_count: usize,
    learning_rate: A,
    eps: A,
}

pub struct Builder<A, K>
where
    K: Kernel,
{
    kernel: std::marker::PhantomData<K>,
    max_iteration_count: Option<usize>,
    learning_rate: Option<A>,
    eps: Option<A>,
}
