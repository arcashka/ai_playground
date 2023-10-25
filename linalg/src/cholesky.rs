use crate::array;

pub struct LMatrix<A> {
    data: array::Array1<A>,
    n: array::Ix,
}

impl<A> LMatrix<A>
where
    A: num_traits::Float,
{
    pub fn inv(&self) -> array::ArrayBase<ndarray::OwnedRepr<A>, array::Ix2> {
        let mut result = array::Array2::<A>::zeros((self.n, self.n));
        for j in 0..self.n {
            // Forward substitution
            let mut y = array::Array1::<A>::zeros(self.n);
            for i in 0..self.n {
                let mut sum = A::zero();
                for k in 0..i {
                    sum = sum + self.l(i, k) * y[k];
                }
                let e_i = if i == j { A::one() } else { A::zero() };
                y[i] = (e_i - sum) / self.l(i, i);
            }

            // Backward substitution
            let mut x = array::Array1::<A>::zeros(self.n);
            for i in (0..self.n).rev() {
                let mut sum = A::zero();
                for k in i + 1..self.n {
                    sum = sum + self.lt(i, k) * x[k];
                }
                x[i] = (y[i] - sum) / self.lt(i, i);
            }
            result.column_mut(j).assign(&x.view());
        }
        result
    }

    #[allow(dead_code)]
    pub fn to_2d_array(&self) -> array::ArrayBase<ndarray::OwnedRepr<A>, array::Ix2>
    where
        A: num_traits::Float,
    {
        let mut result =
            array::ArrayBase::<ndarray::OwnedRepr<A>, array::Ix2>::zeros((self.n, self.n));
        for i in 0..Self::index_to_l(self.n - 1, self.n - 1) + 1 {
            result[Self::index_from_l(i)] = self.data[i];
        }
        result
    }

    fn l(&self, i: usize, j: usize) -> A {
        self.data[Self::index_to_l(i, j)]
    }

    fn lt(&self, i: usize, j: usize) -> A {
        self.data[Self::index_to_l(j, i)]
    }

    fn index_to_l(i: usize, j: usize) -> usize {
        (i * (i + 1) / 2) + j
    }

    fn index_from_l(k: usize) -> (usize, usize) {
        let i = ((-1.0 + f64::sqrt(1.0 + 8.0 * (k as f64))) / 2.0).floor() as usize;
        let j = k - (i * (i + 1)) / 2;
        (i, j)
    }
}

pub trait LLT<S, A>
where
    S: ndarray::Data<Elem = A>,
{
    fn llt(&self) -> LMatrix<A>;
}

impl<S, A> LLT<S, A> for array::ArrayBase<S, array::Ix2>
where
    S: ndarray::Data<Elem = A>,
    A: num_traits::Float,
{
    fn llt(&self) -> LMatrix<A> {
        assert!(self.nrows() == self.ncols());
        let n = self.nrows();
        let mut result = array::Array1::<A>::zeros(n * (n + 1) / 2);
        fn l_index(i: usize, j: usize) -> usize {
            (i * (i + 1) / 2) + j
        }
        for i in 0..n {
            for j in 0..i + 1 {
                let mut sum = A::zero();
                if i == j {
                    for k in 0..i {
                        sum = sum + result[l_index(i, k)].powi(2);
                    }
                    result[l_index(i, i)] = (self[(i, i)] - sum).sqrt();
                } else {
                    for k in 0..j {
                        sum = sum + result[l_index(i, k)] * result[l_index(j, k)];
                    }
                    result[l_index(i, j)] = (self[(i, j)] - sum) / result[l_index(j, j)];
                }
            }
        }
        LMatrix { data: result, n }
    }
}

#[cfg(test)]
mod tests {
    use super::LLT;
    use crate::array;
    use crate::array::Transpose;
    use crate::linalg::Dot;

    #[test]
    fn test_llt_decomposition() {
        let a = array![
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0]
        ];
        let l_matrix = a.llt();
        let l_2d = l_matrix.to_2d_array();
        let reconstructed_a = l_2d.dot(&l_2d.t());
        assert_eq!(a, reconstructed_a);
    }
}
