use crate::array;

pub trait Arithmetic<A, S, D>
where
    S: ndarray::Data<Elem = A>,
    D: ndarray::Dimension,
{
    fn sub(&self, rhs: &array::ArrayBase<S, D>) -> array::Array1<A>;
    fn add(&self, rhs: &array::ArrayBase<S, D>) -> array::Array1<A>;
    fn scaled_add(&self, alpha: A, rhs: &array::ArrayBase<S, D>) -> array::Array1<A>;
}

impl<A, S1, S2> Arithmetic<A, S2, array::Ix1> for array::ArrayBase<S1, array::Ix1>
where
    A: num_traits::Float + std::iter::Sum,
    S1: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = A>,
{
    fn sub(&self, rhs: &array::ArrayBase<S2, array::Ix1>) -> array::Array1<A> {
        self.scaled_add(A::one().neg(), rhs)
    }
    fn add(&self, rhs: &array::ArrayBase<S2, array::Ix1>) -> array::Array1<A> {
        self.scaled_add(A::one(), rhs)
    }
    fn scaled_add(&self, alpha: A, rhs: &array::ArrayBase<S2, array::Ix1>) -> array::Array1<A> {
        scaled_add(alpha, self, rhs)
    }
}

fn scaled_add<A, S1, S2>(
    alpha: A,
    lhs: &array::ArrayBase<S1, array::Ix1>,
    rhs: &array::ArrayBase<S2, array::Ix1>,
) -> array::ArrayBase<ndarray::OwnedRepr<A>, array::Ix1>
where
    S1: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = A>,
    A: num_traits::Float + std::iter::Sum,
{
    lhs.into_iter()
        .zip(rhs.into_iter())
        .map(|(a, b)| *a + (alpha * *b))
        .collect()
}
