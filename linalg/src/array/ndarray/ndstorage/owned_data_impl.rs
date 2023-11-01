use super::{Data, Ownership, Owning};
use crate::LinalgError;

use std::alloc::{alloc, dealloc, realloc, Layout};
use std::ptr;

impl<A, O: Ownership> Drop for Data<A, O> {
    fn drop(&mut self) {
        if std::any::TypeId::of::<O>() == std::any::TypeId::of::<Owning>() {
            let layout = Layout::array::<A>(self.len).unwrap();
            unsafe { dealloc(self.data as *mut u8, layout) };
        }
    }
}

impl<A> Data<A, Owning> {
    pub(super) fn zeros(len: usize) -> Self
    where
        A: num_traits::Zero,
    {
        let layout = Layout::array::<A>(len).unwrap();
        let data = unsafe { alloc(layout) as *mut A };

        for i in 0..len {
            unsafe {
                ptr::write(data.add(i), A::zero());
            }
        }
        Self {
            data,
            len,
            ownership: std::marker::PhantomData,
        }
    }

    pub(super) fn from_vec(vec: Vec<A>) -> Self {
        let boxed_slice = vec.into_boxed_slice();
        let data = Box::into_raw(boxed_slice) as *mut A;

        Self {
            data,
            len: vec.len(),
            ownership: std::marker::PhantomData,
        }
    }

    pub(super) fn append(&mut self, vec: Vec<A>) -> Result<(), LinalgError> {
        let old_layout = Layout::array::<A>(self.len).unwrap();
        let new_len = self.len + vec.len();
        let new_layout = Layout::array::<A>(new_len).unwrap();

        let new_data =
            unsafe { realloc(self.data as *mut _, old_layout, new_layout.size()) as *mut A };
        if new_data.is_null() {
            return Err(LinalgError::AllocationError);
        }

        unsafe {
            ptr::copy_nonoverlapping(vec.as_ptr(), new_data.add(self.len), vec.len());
        }

        self.len = new_len;
        self.data = new_data;
        Ok(())
    }
}
