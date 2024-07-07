use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector<const N: usize> {
    data: [f32; N],
}

impl<const N: usize> Vector<N> {
    pub fn zeroed() -> Self {
        Self { data: [0.; N] }
    }

    pub fn splat(x: f32) -> Self {
        Self { data: [x; N] }
    }

    pub fn len(&self) -> usize {
        N
    }
}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const N: usize> IntoIterator for Vector<N> {
    type Item = f32;

    type IntoIter = std::array::IntoIter<f32, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<const N: usize> From<[f32; N]> for Vector<N> {
    fn from(value: [f32; N]) -> Self {
        Self { data: value }
    }
}
