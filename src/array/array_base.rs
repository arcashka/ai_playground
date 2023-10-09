use crate::array::types::*;

use std::fmt;
use std::ops::{Index, IndexMut};

pub struct ArrayBase<S, D>
where
    S: ndarray::RawData,
{
    a: ndarray::ArrayBase<S, D>,
}

type ViewRepr<A> = ndarray::ViewRepr<A>;
type OwnedRepr<A> = ndarray::OwnedRepr<A>;

pub type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;
pub type ArrayView1<'a, S> = ArrayView<'a, S, Ix1>;
pub type ArrayView2<'a, S> = ArrayView<'a, S, Ix2>;
pub type ArrayViewMut<'a, A, D> = ArrayBase<ViewRepr<&'a mut A>, D>;
pub type ArrayViewMut1<'a, A> = ArrayViewMut<'a, A, Ix1>;

pub type Array<A, D> = ArrayBase<OwnedRepr<A>, D>;
pub type Array1<A> = Array<A, Ix1>;
pub type Array2<A> = Array<A, Ix2>;

#[macro_export]
macro_rules! array {
    ($($args:tt)*) => {
        {
            let ndarray_array = ndarray::array![$($args)*];
            crate::array::ArrayBase::new(ndarray_array)
        }
    };
}

impl<A> Array<A, Ix1> {
    pub fn from_vec(v: Vec<A>) -> Self {
        Self {
            a: ndarray::Array::<A, Ix1>::from_vec(v),
        }
    }
}

impl<A, S, D> ArrayBase<S, D>
where
    D: ndarray::Dimension,
    S: ndarray::RawData<Elem = A>,
{
    // needed for array! macro to work
    pub fn new(a: ndarray::ArrayBase<S, D>) -> Self {
        Self { a }
    }

    pub fn len(&self) -> Ix {
        self.a.len()
    }

    pub fn len_of(&self, axis: Axis) -> usize {
        self.a.len_of(axis)
    }

    pub fn dim(&self) -> D::Pattern {
        self.a.dim()
    }

    pub fn strides(&self) -> &[isize] {
        self.a.strides()
    }

    pub fn shape(&self) -> &[usize] {
        self.a.shape()
    }

    pub fn into_shape<E>(self, shape: E) -> Result<ArrayBase<S, E::Dim>, ShapeError>
    where
        E: ndarray::IntoDimension,
    {
        self.a.into_shape(shape).map(|array| ArrayBase { a: array })
    }

    pub fn assign<E: ndarray::Dimension, S2>(&mut self, rhs: &ArrayBase<S2, E>)
    where
        S: ndarray::DataMut,
        A: Clone,
        S2: ndarray::Data<Elem = A>,
    {
        self.a.assign(&rhs.a)
    }

    pub(crate) fn inner_impl(&self) -> &ndarray::ArrayBase<S, D> {
        &self.a
    }

    pub(crate) fn inner_impl_mut(&mut self) -> &mut ndarray::ArrayBase<S, D> {
        &mut self.a
    }
}

impl<A, S> ArrayBase<S, Ix2>
where
    S: ndarray::Data<Elem = A>,
{
    pub fn row(&self, index: ndarray::Ix) -> ArrayView<'_, A, Ix1> {
        ArrayView {
            a: self.a.row(index),
        }
    }

    pub fn ncols(&self) -> usize {
        self.a.ncols()
    }

    pub fn nrows(&self) -> usize {
        self.a.nrows()
    }
}

impl<A, S> ArrayBase<S, Ix2>
where
    S: ndarray::DataMut<Elem = A>,
{
    pub fn row_mut(&mut self, index: Ix) -> ArrayViewMut1<'_, A> {
        ArrayViewMut1::<'_, A> {
            a: self
                .inner_impl_mut()
                .index_axis_mut(ndarray::Axis(0), index),
        }
    }

    pub fn column_mut(&mut self, index: Ix) -> ArrayViewMut1<'_, A> {
        ArrayViewMut1::<'_, A> {
            a: self
                .inner_impl_mut()
                .index_axis_mut(ndarray::Axis(1), index),
        }
    }
}

impl<A, S, D> ArrayBase<S, D>
where
    S: ndarray::DataOwned<Elem = A>,
    D: ndarray::Dimension,
{
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: num_traits::Zero + Clone,
        Sh: ndarray::ShapeBuilder<Dim = D>,
    {
        Self {
            a: ndarray::ArrayBase::<S, D>::zeros(shape),
        }
    }

    pub fn from_shape_vec<Sh>(shape: Sh, v: Vec<A>) -> Result<Self, ShapeError>
    where
        Sh: ndarray::ShapeBuilder<Dim = D>,
    {
        let array = ndarray::ArrayBase::from_shape_vec(shape, v)?;
        Ok(Self { a: array })
    }

    pub fn view(&self) -> ArrayView<'_, A, D> {
        ArrayView {
            a: self.inner_impl().view(),
        }
    }
}

impl<A> ArrayBase<ndarray::OwnedRepr<A>, Ix2> {
    pub fn push_row(&mut self, row: ArrayView<A, Ix1>) -> Result<(), ShapeError>
    where
        A: Clone,
    {
        self.a.push_row(row.a)?;
        Ok(())
    }
}

impl<S, D, I> Index<I> for ArrayBase<S, D>
where
    D: ndarray::Dimension,
    I: ndarray::NdIndex<D>,
    S: ndarray::Data,
{
    type Output = S::Elem;
    #[inline]
    fn index(&self, index: I) -> &S::Elem {
        &self.a[index]
    }
}

impl<S, D, I> IndexMut<I> for ArrayBase<S, D>
where
    D: ndarray::Dimension,
    I: ndarray::NdIndex<D>,
    S: ndarray::DataMut,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut S::Elem {
        self.inner_impl_mut().index_mut(index)
    }
}

impl<S: ndarray::RawDataClone, D: Clone> Clone for ArrayBase<S, D> {
    fn clone(&self) -> ArrayBase<S, D> {
        Self { a: self.a.clone() }
    }

    fn clone_from(&mut self, other: &Self) {
        self.a.clone_from(&other.a);
    }
}

impl<S: ndarray::RawDataClone + Copy, D: Copy> Copy for ArrayBase<S, D> {}

impl<A: fmt::Debug, S, D: ndarray::Dimension> fmt::Debug for ArrayBase<S, D>
where
    S: ndarray::Data<Elem = A>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.a.fmt(f)
    }
}

impl<A, B, S, S2, D> PartialEq<ArrayBase<S2, D>> for ArrayBase<S, D>
where
    A: PartialEq<B>,
    S: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = B>,
    D: ndarray::Dimension,
{
    fn eq(&self, rhs: &ArrayBase<S2, D>) -> bool {
        self.a.eq(&rhs.a)
    }
}

pub trait Transpose<A, S>
where
    S: ndarray::Data<Elem = A>,
{
    fn t(&self) -> ArrayView<'_, A, Ix2>;
}

impl<A, S> Transpose<A, S> for ArrayBase<S, Ix2>
where
    S: ndarray::Data<Elem = A>,
{
    fn t(&self) -> ArrayView<'_, A, Ix2> {
        ArrayView::<'_, A, Ix2> { a: self.a.t() }
    }
}

impl<A, S> FromIterator<A> for ArrayBase<S, Ix1>
where
    S: ndarray::DataOwned<Elem = A>,
{
    fn from_iter<I>(iterable: I) -> Self
    where
        I: IntoIterator<Item = A>,
    {
        Self {
            a: ndarray::ArrayBase::<S, Ix1>::from_iter(iterable),
        }
    }
}

impl<'a, S, D> IntoIterator for &'a ArrayBase<S, D>
where
    D: ndarray::Dimension,
    S: ndarray::Data,
{
    type Item = &'a S::Elem;
    type IntoIter = ndarray::iter::Iter<'a, S::Elem, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.a.iter()
    }
}

impl<'a, S, D> IntoIterator for &'a mut ArrayBase<S, D>
where
    D: ndarray::Dimension,
    S: ndarray::DataMut,
{
    type Item = &'a mut S::Elem;
    type IntoIter = ndarray::iter::IterMut<'a, S::Elem, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.a.iter_mut()
    }
}

impl<'a, A, D> IntoIterator for ArrayView<'a, A, D>
where
    D: ndarray::Dimension,
{
    type Item = &'a A;
    type IntoIter = ndarray::iter::Iter<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.a.into_iter()
    }
}

impl<'a, A, D> IntoIterator for ArrayViewMut<'a, A, D>
where
    D: ndarray::Dimension,
{
    type Item = &'a mut A;
    type IntoIter = ndarray::iter::IterMut<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.a.into_iter()
    }
}
