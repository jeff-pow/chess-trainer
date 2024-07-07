use crate::vector::Vector;
use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix<const N: usize, const M: usize> {
    data: [Vector<N>; M],
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    pub fn zeroed() -> Self {
        Self {
            data: [Vector::<N>::zeroed(); M],
        }
    }

    pub fn splat(x: f32) -> Self {
        Self {
            data: [Vector::<N>::splat(x); M],
        }
    }

    pub fn rows(&self) -> usize {
        N
    }

    pub fn cols(&self) -> usize {
        M
    }

    /// Returns a tuple of (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (N, M)
    }

    pub fn iterrows(&self) -> impl Iterator<Item = &Vector<N>> {
        self.data.iter()
    }

    pub fn transpose(&mut self) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows {
            for j in i + 1..cols {
                let tmp = self[i][j];
                self[i][j] = self[j][i];
                self[j][i] = tmp;
            }
        }
    }
}

impl<const N: usize, const M: usize> Index<usize> for Matrix<N, M> {
    type Output = Vector<N>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize, const M: usize> IndexMut<usize> for Matrix<N, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const N: usize, const M: usize> From<[[f32; N]; M]> for Matrix<N, M> {
    fn from(value: [[f32; N]; M]) -> Self {
        let mut data = [Vector::from([0.0; N]); M];
        for (i, row) in value.iter().enumerate() {
            data[i] = Vector::from(*row);
        }
        Self { data }
    }
}

#[cfg(test)]
mod bruh {
    use crate::matrix::Matrix;

    #[test]
    fn test_transpose() {
        let mut matrix: Matrix<3, 3> = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]].into();
        matrix.transpose();

        let expected_transpose: Matrix<3, 3> = [[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]].into();
        assert_eq!(matrix, expected_transpose);
    }
}
