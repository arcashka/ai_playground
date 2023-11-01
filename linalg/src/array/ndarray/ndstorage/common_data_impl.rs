use super::{Data, Ownership};

use std::alloc::{alloc, Layout};
use std::fmt;
use std::slice;

impl<A, O: Ownership> Data<A, O> {
    pub unsafe fn get(&self, index: usize) -> A {
        *self.data.add(index)
    }
}

impl<A: Clone, O: Ownership> Clone for Data<A, O> {
    fn clone(&self) -> Self {
        let layout = Layout::array::<A>(self.len).unwrap();
        let new_data = unsafe { alloc(layout) as *mut A };
        for i in 0..self.len {
            let val = unsafe { self.data.add(i).read() };
            unsafe { new_data.add(i).write(val.clone()) };
        }

        Self {
            data: new_data,
            len: self.len,
            ownership: std::marker::PhantomData,
        }
    }
}

impl<A: fmt::Debug, O: Ownership> fmt::Debug for Data<A, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slice = unsafe { slice::from_raw_parts(self.data, self.len) };
        f.debug_tuple("NDData").field(&slice).finish()
    }
}
