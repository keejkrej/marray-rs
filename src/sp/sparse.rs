use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Coordinate list (COO) sparse matrix format.
#[derive(Clone, Debug, PartialEq)]
pub struct COOMatrix {
    pub data: Vec<f64>,
    pub row: Vec<usize>,
    pub col: Vec<usize>,
    pub shape: (usize, usize),
}

impl COOMatrix {
    pub fn new(data: Vec<f64>, row: Vec<usize>, col: Vec<usize>, shape: (usize, usize)) -> Self {
        assert_eq!(data.len(), row.len());
        assert_eq!(data.len(), col.len());
        Self {
            data,
            row,
            col,
            shape,
        }
    }

    pub fn from_dense(mat: &Array2<f64>) -> Self {
        let (nrows, ncols) = mat.dim();
        let mut data = Vec::new();
        let mut row = Vec::new();
        let mut col = Vec::new();

        for i in 0..nrows {
            for j in 0..ncols {
                let val = mat[[i, j]];
                if val.abs() > 1e-15 {
                    data.push(val);
                    row.push(i);
                    col.push(j);
                }
            }
        }

        Self {
            data,
            row,
            col,
            shape: (nrows, ncols),
        }
    }

    pub fn to_dense(&self) -> Array2<f64> {
        let mut mat = Array2::zeros(self.shape);
        for k in 0..self.data.len() {
            mat[[self.row[k], self.col[k]]] += self.data[k];
        }
        mat
    }

    pub fn to_csr(&self) -> CSRMatrix {
        let (nrows, ncols) = self.shape;
        let nnz = self.data.len();

        let mut row_counts = vec![0usize; nrows];
        for &r in &self.row {
            row_counts[r] += 1;
        }

        let mut indptr = vec![0usize; nrows + 1];
        for i in 0..nrows {
            indptr[i + 1] = indptr[i] + row_counts[i];
        }

        let mut next_pos = indptr.clone();
        let mut indices = vec![0usize; nnz];
        let mut data = vec![0.0f64; nnz];

        for k in 0..nnz {
            let r = self.row[k];
            let pos = next_pos[r];
            indices[pos] = self.col[k];
            data[pos] = self.data[k];
            next_pos[r] += 1;
        }

        CSRMatrix {
            data,
            indices,
            indptr,
            shape: (nrows, ncols),
        }
    }

    pub fn to_csc(&self) -> CSCMatrix {
        self.to_csr().to_csc()
    }
}

/// Compressed Sparse Row (CSR) matrix format.
#[derive(Clone, Debug, PartialEq)]
pub struct CSRMatrix {
    pub data: Vec<f64>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
    pub shape: (usize, usize),
}

impl CSRMatrix {
    pub fn new(
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: (usize, usize),
    ) -> Self {
        assert_eq!(data.len(), indices.len());
        assert_eq!(indptr.len(), shape.0 + 1);
        Self {
            data,
            indices,
            indptr,
            shape,
        }
    }

    pub fn from_dense(mat: &Array2<f64>) -> Self {
        COOMatrix::from_dense(mat).to_csr()
    }

    pub fn to_dense(&self) -> Array2<f64> {
        let mut mat = Array2::zeros(self.shape);
        let (nrows, _) = self.shape;

        for i in 0..nrows {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            for k in start..end {
                mat[[i, self.indices[k]]] += self.data[k];
            }
        }
        mat
    }

    pub fn to_coo(&self) -> COOMatrix {
        let mut row = Vec::with_capacity(self.data.len());
        for i in 0..self.shape.0 {
            for _ in self.indptr[i]..self.indptr[i + 1] {
                row.push(i);
            }
        }
        COOMatrix {
            data: self.data.clone(),
            row,
            col: self.indices.clone(),
            shape: self.shape,
        }
    }

    pub fn to_csc(&self) -> CSCMatrix {
        let (nrows, ncols) = self.shape;
        let nnz = self.data.len();

        let mut col_counts = vec![0usize; ncols];
        for &c in &self.indices {
            col_counts[c] += 1;
        }

        let mut indptr = vec![0usize; ncols + 1];
        for j in 0..ncols {
            indptr[j + 1] = indptr[j] + col_counts[j];
        }

        let mut next_pos = indptr.clone();
        let mut row_indices = vec![0usize; nnz];
        let mut data = vec![0.0f64; nnz];

        for i in 0..nrows {
            for pos in self.indptr[i]..self.indptr[i + 1] {
                let col = self.indices[pos];
                let dest = next_pos[col];
                row_indices[dest] = i;
                data[dest] = self.data[pos];
                next_pos[col] += 1;
            }
        }

        CSCMatrix {
            data,
            indices: row_indices,
            indptr,
            shape: (nrows, ncols),
        }
    }

    pub fn transpose(&self) -> CSRMatrix {
        self.to_csc().to_csr_format()
    }

    /// Sparse matrix-vector product: $y = A x$.
    pub fn dot_vec(&self, x: &Array1<f64>) -> Array1<f64> {
        let (nrows, ncols) = self.shape;
        assert_eq!(x.len(), ncols, "Vector length must match matrix columns");
        let mut y = Array1::<f64>::zeros(nrows);

        for i in 0..nrows {
            let mut sum = 0.0;
            for k in self.indptr[i]..self.indptr[i + 1] {
                sum += self.data[k] * x[self.indices[k]];
            }
            y[i] = sum;
        }
        y
    }

    /// Sparse matrix-matrix multiplication: $C = A B$.
    pub fn dot_mat(&self, other: &CSRMatrix) -> CSRMatrix {
        let (m, k1) = self.shape;
        let (k2, n) = other.shape;
        assert_eq!(k1, k2, "Inner dimensions must agree");

        let mut row_entries = HashMap::new();
        let mut c_data = Vec::new();
        let mut c_indices = Vec::new();
        let mut c_indptr = vec![0usize; m + 1];

        for i in 0..m {
            row_entries.clear();
            for a_idx in self.indptr[i]..self.indptr[i + 1] {
                let a_col = self.indices[a_idx];
                let a_val = self.data[a_idx];

                for b_idx in other.indptr[a_col]..other.indptr[a_col + 1] {
                    let b_col = other.indices[b_idx];
                    let b_val = other.data[b_idx];
                    *row_entries.entry(b_col).or_insert(0.0) += a_val * b_val;
                }
            }

            let mut sorted_cols: Vec<_> = row_entries
                .iter()
                .filter(|&(_, v)| v.abs() > 1e-15)
                .map(|(&c, &v)| (c, v))
                .collect();
            sorted_cols.sort_by_key(|&(c, _)| c);

            for (c, v) in sorted_cols {
                c_indices.push(c);
                c_data.push(v);
            }
            c_indptr[i + 1] = c_data.len();
        }

        CSRMatrix {
            data: c_data,
            indices: c_indices,
            indptr: c_indptr,
            shape: (m, n),
        }
    }

    /// Elementwise addition: $A + B$.
    pub fn add(&self, other: &CSRMatrix) -> CSRMatrix {
        assert_eq!(self.shape, other.shape);
        let (nrows, ncols) = self.shape;

        let mut out_data = Vec::new();
        let mut out_indices = Vec::new();
        let mut out_indptr = vec![0usize; nrows + 1];

        for i in 0..nrows {
            let mut ptr_a = self.indptr[i];
            let end_a = self.indptr[i + 1];
            let mut ptr_b = other.indptr[i];
            let end_b = other.indptr[i + 1];

            while ptr_a < end_a || ptr_b < end_b {
                if ptr_a < end_a && (ptr_b == end_b || self.indices[ptr_a] < other.indices[ptr_b]) {
                    out_indices.push(self.indices[ptr_a]);
                    out_data.push(self.data[ptr_a]);
                    ptr_a += 1;
                } else if ptr_b < end_b && (ptr_a == end_a || other.indices[ptr_b] < self.indices[ptr_a]) {
                    out_indices.push(other.indices[ptr_b]);
                    out_data.push(other.data[ptr_b]);
                    ptr_b += 1;
                } else {
                    let sum = self.data[ptr_a] + other.data[ptr_b];
                    if sum.abs() > 1e-15 {
                        out_indices.push(self.indices[ptr_a]);
                        out_data.push(sum);
                    }
                    ptr_a += 1;
                    ptr_b += 1;
                }
            }
            out_indptr[i + 1] = out_data.len();
        }

        CSRMatrix {
            data: out_data,
            indices: out_indices,
            indptr: out_indptr,
            shape: (nrows, ncols),
        }
    }

    /// Elementwise subtraction: $A - B$.
    pub fn sub(&self, other: &CSRMatrix) -> CSRMatrix {
        let neg_other = other.scale(-1.0);
        self.add(&neg_other)
    }

    /// Scalar multiplication: $\alpha A$.
    pub fn scale(&self, scalar: f64) -> CSRMatrix {
        let data: Vec<f64> = self.data.iter().map(|&v| v * scalar).collect();
        CSRMatrix {
            data,
            indices: self.indices.clone(),
            indptr: self.indptr.clone(),
            shape: self.shape,
        }
    }
}

/// Compressed Sparse Column (CSC) matrix format.
#[derive(Clone, Debug, PartialEq)]
pub struct CSCMatrix {
    pub data: Vec<f64>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
    pub shape: (usize, usize),
}

impl CSCMatrix {
    pub fn new(
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: (usize, usize),
    ) -> Self {
        assert_eq!(data.len(), indices.len());
        assert_eq!(indptr.len(), shape.1 + 1);
        Self {
            data,
            indices,
            indptr,
            shape,
        }
    }

    pub fn from_dense(mat: &Array2<f64>) -> Self {
        CSRMatrix::from_dense(mat).to_csc()
    }

    pub fn to_dense(&self) -> Array2<f64> {
        let mut mat = Array2::zeros(self.shape);
        let (_, ncols) = self.shape;

        for j in 0..ncols {
            for k in self.indptr[j]..self.indptr[j + 1] {
                mat[[self.indices[k], j]] += self.data[k];
            }
        }
        mat
    }

    pub fn to_csr(&self) -> CSRMatrix {
        self.to_csr_format()
    }

    fn to_csr_format(&self) -> CSRMatrix {
        let (nrows, ncols) = self.shape;
        let nnz = self.data.len();

        let mut row_counts = vec![0usize; nrows];
        for &r in &self.indices {
            row_counts[r] += 1;
        }

        let mut indptr = vec![0usize; nrows + 1];
        for i in 0..nrows {
            indptr[i + 1] = indptr[i] + row_counts[i];
        }

        let mut next_pos = indptr.clone();
        let mut col_indices = vec![0usize; nnz];
        let mut data = vec![0.0f64; nnz];

        for j in 0..ncols {
            for pos in self.indptr[j]..self.indptr[j + 1] {
                let row = self.indices[pos];
                let dest = next_pos[row];
                col_indices[dest] = j;
                data[dest] = self.data[pos];
                next_pos[row] += 1;
            }
        }

        CSRMatrix {
            data,
            indices: col_indices,
            indptr,
            shape: (nrows, ncols),
        }
    }

    pub fn to_coo(&self) -> COOMatrix {
        let mut col = Vec::with_capacity(self.data.len());
        for j in 0..self.shape.1 {
            for _ in self.indptr[j]..self.indptr[j + 1] {
                col.push(j);
            }
        }
        COOMatrix {
            data: self.data.clone(),
            row: self.indices.clone(),
            col,
            shape: self.shape,
        }
    }
}

// --- Sparse Linear Algebra ---

pub mod linalg {
    use super::*;

    /// Compute the matrix norm of a sparse matrix.
    pub fn norm(a: &CSRMatrix, ord: Option<&str>) -> f64 {
        match ord.unwrap_or("fro") {
            "fro" => a.data.iter().map(|&v| v * v).sum::<f64>().sqrt(),
            "1" => {
                // Max column sum
                let mut col_sums = vec![0.0; a.shape.1];
                for (k, &c) in a.indices.iter().enumerate() {
                    col_sums[c] += a.data[k].abs();
                }
                col_sums.into_iter().fold(0.0_f64, f64::max)
            }
            "inf" => {
                // Max row sum
                let mut max_sum = 0.0;
                for i in 0..a.shape.0 {
                    let mut sum = 0.0;
                    for k in a.indptr[i]..a.indptr[i + 1] {
                        sum += a.data[k].abs();
                    }
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                max_sum
            }
            _ => a.data.iter().map(|&v| v * v).sum::<f64>().sqrt(),
        }
    }

    /// Solve linear system $A x = b$ using BiCGSTAB (Biconjugate Gradient Stabilized).
    pub fn spsolve(a: &CSRMatrix, b: &Array1<f64>) -> Result<Array1<f64>, String> {
        let (n, m) = a.shape;
        if n != m {
            return Err("Matrix must be square".to_string());
        }
        if b.len() != n {
            return Err("Dimension mismatch".to_string());
        }

        let b_norm = b.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if b_norm < 1e-15 {
            return Ok(Array1::zeros(n));
        }

        let mut x = Array1::<f64>::zeros(n);
        let mut r = b - &a.dot_vec(&x);
        let r0_hat = r.clone();

        let mut rho_prev = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;

        let mut v = Array1::<f64>::zeros(n);
        let mut p = Array1::<f64>::zeros(n);

        let max_iter = n * 2 + 100;
        let tol = 1e-10;

        for _ in 0..max_iter {
            let rho = r0_hat.iter().zip(r.iter()).map(|(&u, &v)| u * v).sum::<f64>();
            if rho.abs() < 1e-20 {
                return Err("BiCGSTAB breakdown (rho=0)".to_string());
            }

            let beta = (rho / rho_prev) * (alpha / omega);
            p = &r + &(&(&p - &(&v * omega)) * beta);

            v = a.dot_vec(&p);
            let denom = r0_hat.iter().zip(v.iter()).map(|(&u, &w)| u * w).sum::<f64>();
            if denom.abs() < 1e-20 {
                return Err("BiCGSTAB breakdown (denom=0)".to_string());
            }
            alpha = rho / denom;

            let s = &r - &(&v * alpha);
            let s_norm = s.iter().map(|&w| w * w).sum::<f64>().sqrt();
            if s_norm / b_norm < tol {
                x = &x + &(&p * alpha);
                return Ok(x);
            }

            let t = a.dot_vec(&s);
            let tt = t.iter().map(|&w| w * w).sum::<f64>();
            if tt < 1e-20 {
                x = &x + &(&p * alpha);
                return Ok(x);
            }
            omega = t.iter().zip(s.iter()).map(|(&u, &w)| u * w).sum::<f64>() / tt;

            x = &x + &(&(&p * alpha) + &(&s * omega));
            r = &s - &(&t * omega);

            let res_norm = r.iter().map(|&w| w * w).sum::<f64>().sqrt();
            if res_norm / b_norm < tol {
                return Ok(x);
            }

            if omega.abs() < 1e-20 {
                break;
            }

            rho_prev = rho;
        }

        Ok(x)
    }

    /// Compute the `k` largest eigenvalues and eigenvectors of a symmetric sparse matrix using power/Lanczos iterations.
    pub fn eigsh(a: &CSRMatrix, k: usize) -> Result<(Array1<f64>, Array2<f64>), String> {
        let (n, m) = a.shape;
        if n != m {
            return Err("Matrix must be square".to_string());
        }
        if k == 0 || k > n {
            return Err("Invalid k".to_string());
        }

        let mut eigenvalues = Vec::with_capacity(k);
        let mut eigenvectors = Array2::<f64>::zeros((n, k));

        let mut a_deflated = a.clone();

        for comp in 0..k {
            // Power iteration with Rayleigh quotient
            let mut v = Array1::from_shape_fn(n, |i| if i % 2 == 0 { 1.0 } else { -1.0 });
            let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            v /= norm;

            let mut lambda = 0.0;
            for _ in 0..300 {
                let w = a_deflated.dot_vec(&v);
                let w_norm = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if w_norm < 1e-15 {
                    break;
                }
                v = &w / w_norm;
                let av = a_deflated.dot_vec(&v);
                lambda = v.iter().zip(av.iter()).map(|(&x, &y)| x * y).sum::<f64>();
            }

            eigenvalues.push(lambda);
            for i in 0..n {
                eigenvectors[[i, comp]] = v[i];
            }

            // Deflate: A_new = A - lambda * v * v^T
            let mut vvt_data = Vec::new();
            let mut vvt_indices = Vec::new();
            let mut vvt_indptr = vec![0usize; n + 1];

            for i in 0..n {
                for j in 0..n {
                    let val = -lambda * v[i] * v[j];
                    if val.abs() > 1e-15 {
                        vvt_indices.push(j);
                        vvt_data.push(val);
                    }
                }
                vvt_indptr[i + 1] = vvt_data.len();
            }

            let vvt = CSRMatrix {
                data: vvt_data,
                indices: vvt_indices,
                indptr: vvt_indptr,
                shape: (n, n),
            };

            a_deflated = a_deflated.add(&vvt);
        }

        Ok((Array1::from_vec(eigenvalues), eigenvectors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sparse_conversions() {
        let dense = array![
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [4.0, 5.0, 6.0]
        ];

        let coo = COOMatrix::from_dense(&dense);
        assert_eq!(coo.data.len(), 6);
        assert_eq!(coo.to_dense(), dense);

        let csr = coo.to_csr();
        assert_eq!(csr.to_dense(), dense);

        let csc = csr.to_csc();
        assert_eq!(csc.to_dense(), dense);

        let roundtrip_csr = csc.to_csr();
        assert_eq!(roundtrip_csr.to_dense(), dense);
    }

    #[test]
    fn test_sparse_arithmetic() {
        let a_dense = array![[1.0, 2.0], [0.0, 3.0]];
        let b_dense = array![[0.0, 1.0], [4.0, 5.0]];

        let a = CSRMatrix::from_dense(&a_dense);
        let b = CSRMatrix::from_dense(&b_dense);

        let c_add = a.add(&b);
        assert_eq!(c_add.to_dense(), array![[1.0, 3.0], [4.0, 8.0]]);

        let c_sub = a.sub(&b);
        assert_eq!(c_sub.to_dense(), array![[1.0, 1.0], [-4.0, -2.0]]);

        let c_dot = a.dot_mat(&b);
        assert_eq!(c_dot.to_dense(), a_dense.dot(&b_dense));

        let x = array![2.0, 3.0];
        let y = a.dot_vec(&x);
        assert_eq!(y, array![8.0, 9.0]);
    }

    #[test]
    fn test_sparse_linalg() {
        // [4  1] [x1] = [1]
        // [1  3] [x2]   [2]
        // 4x1 + x2 = 1, x1 + 3x2 = 2 => 11x2 = 7 => x2 = 7/11, x1 = 1/11
        let a = CSRMatrix::from_dense(&array![[4.0, 1.0], [1.0, 3.0]]);
        let b = array![1.0, 2.0];

        let x = linalg::spsolve(&a, &b).unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-6);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-6);

        let frob = linalg::norm(&a, Some("fro"));
        let exact_frob = (16.0 + 1.0 + 1.0 + 9.0_f64).sqrt();
        assert!((frob - exact_frob).abs() < 1e-9);

        // Eigenvalues
        let (eigvals, _) = linalg::eigsh(&a, 1).unwrap();
        // Characteristic eq: (4 - L)(3 - L) - 1 = L^2 - 7L + 11 = 0
        // L = (7 + sqrt(49 - 44))/2 = (7 + sqrt(5))/2 ≈ 4.6180339887
        let exact_eig = (7.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((eigvals[0] - exact_eig).abs() < 1e-3);
    }
}
