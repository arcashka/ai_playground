use crate::array;

pub trait Dot<Rhs> {
    type Output;
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

type Ix1 = array::Ix1;
type Ix2 = array::Ix2;

impl<A, S, S2> Dot<array::ArrayBase<S2, Ix1>> for array::ArrayBase<S, Ix1>
where
    S: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = A>,
    A: num_traits::Float,
{
    type Output = A;

    fn dot(&self, rhs: &array::ArrayBase<S2, Ix1>) -> A {
        let mut result = A::zero();
        for i in 0..self.len() {
            result = result + (self[i] * rhs[i]);
        }
        result
    }
}

impl<A, S, S2> Dot<array::ArrayBase<S2, Ix2>> for array::ArrayBase<S, Ix2>
where
    S: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = A>,
    A: num_traits::Float + num_traits::Zero,
{
    type Output = array::Array2<A>;

    fn dot(&self, rhs: &array::ArrayBase<S2, Ix2>) -> array::Array2<A> {
        assert!(self.ncols() == rhs.nrows());
        let mut result = array::Array2::<A>::zeros((self.nrows(), rhs.ncols()));
        for i in 0..self.nrows() {
            for j in 0..rhs.ncols() {
                let mut sum = A::zero();
                for k in 0..self.ncols() {
                    sum = sum + self[(i, k)] * rhs[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Dot;
    use crate::array;
    use crate::array::Transpose;

    #[test]
    fn test_vector_vector_dot() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let result = a.dot(&b);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_matrix_matrix_dot() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let result = a.dot(&b);
        assert_eq!(result, array![[19.0, 22.0], [43.0, 50.0]]);
    }

    #[test]
    fn test_row_vector_matrix_dot() {
        let a = array![[1.0, 2.0]];
        let b = array![[3.0, 4.0], [5.0, 6.0]];
        let result = a.dot(&b);
        assert_eq!(result, array![[13.0, 16.0]]);
    }

    #[test]
    fn test_matrix_column_vector_dot() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0]];
        let result = a.dot(&b.t());

        let expected_result = array![[17.0, 39.0]];
        assert_eq!(result, expected_result.t());
    }
}
