use ndarray::{Array1, Array2};

fn ndarray_to_nalgebra(arr: &Array2<f64>) -> nalgebra::DMatrix<f64> {
    let (nrows, ncols) = arr.dim();
    nalgebra::DMatrix::from_row_iterator(nrows, ncols, arr.iter().cloned())
}

fn nalgebra_to_ndarray(mat: nalgebra::DMatrix<f64>) -> Array2<f64> {
    let (nrows, ncols) = mat.shape();
    let mut flat = Vec::with_capacity(nrows * ncols);
    for row in mat.row_iter() {
        flat.extend(row.iter().cloned());
    }
    Array2::from_shape_vec((nrows, ncols), flat).unwrap()
}

pub struct SVDResult {
    pub u: Option<Array2<f64>>,
    pub s: Array1<f64>,
    pub vh: Option<Array2<f64>>,
}

/// Inverse of a matrix.
pub fn inv(arr: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mat = ndarray_to_nalgebra(arr);
    let inv_mat = mat.try_inverse().ok_or_else(|| "Matrix is singular".to_string())?;
    Ok(nalgebra_to_ndarray(inv_mat))
}

/// Determinant of a matrix.
pub fn det(arr: &Array2<f64>) -> f64 {
    let mat = ndarray_to_nalgebra(arr);
    mat.determinant()
}

/// Solve a linear matrix equation, or system of linear scalar equations.
pub fn solve(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
    let a_mat = ndarray_to_nalgebra(a);
    let b_mat = ndarray_to_nalgebra(b);
    let decomp = a_mat.lu();
    let x_mat = decomp.solve(&b_mat).ok_or_else(|| "Matrix solver failed".to_string())?;
    Ok(nalgebra_to_ndarray(x_mat))
}

/// Solve a linear system with a 1D vector target.
pub fn solve_vec(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let a_mat = ndarray_to_nalgebra(a);
    let b_vec = nalgebra::DVector::from_iterator(b.len(), b.iter().cloned());
    let decomp = a_mat.lu();
    let x_vec = decomp.solve(&b_vec).ok_or_else(|| "Matrix solver failed".to_string())?;
    Ok(Array1::from_vec(x_vec.iter().cloned().collect()))
}

/// Cholesky decomposition.
pub fn cholesky(arr: &Array2<f64>, lower: bool) -> Result<Array2<f64>, String> {
    let mat = ndarray_to_nalgebra(arr);
    let chol = mat.cholesky().ok_or_else(|| "Matrix is not positive-definite".to_string())?;
    let l_mat = chol.unpack();
    if lower {
        Ok(nalgebra_to_ndarray(l_mat))
    } else {
        Ok(nalgebra_to_ndarray(l_mat.transpose()))
    }
}

/// Singular Value Decomposition.
pub fn svd(arr: &Array2<f64>) -> SVDResult {
    let mat = ndarray_to_nalgebra(arr);
    let svd = mat.svd(true, true);
    let u = svd.u.map(|u_mat| nalgebra_to_ndarray(u_mat));
    let s = Array1::from_vec(svd.singular_values.iter().cloned().collect());
    let vh = svd.v_t.map(|vt_mat| nalgebra_to_ndarray(vt_mat));
    SVDResult { u, s, vh }
}

/// Compute singular values of a matrix.
pub fn svdvals(arr: &Array2<f64>) -> Array1<f64> {
    let mat = ndarray_to_nalgebra(arr);
    let svd = mat.svd(false, false);
    Array1::from_vec(svd.singular_values.iter().cloned().collect())
}

/// Eigenvalue decomposition of a general real matrix.
/// Returns (eigenvalues as column vector, eigenvectors as columns).
pub fn eig(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err("Matrix must be square".to_string());
    }
    let n = nrows;
    if n == 0 {
        return Err("Matrix must be non-empty".to_string());
    }
    if n == 1 {
        let eigvals = Array2::from_shape_vec((1, 1), vec![a[[0, 0]]]).unwrap();
        let eigvecs = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        return Ok((eigvals, eigvecs));
    }
    if n == 2 {
        let a00 = a[[0, 0]];
        let a01 = a[[0, 1]];
        let a10 = a[[1, 0]];
        let a11 = a[[1, 1]];
        let tr = a00 + a11;
        let det = a00 * a11 - a01 * a10;
        let disc = tr * tr - 4.0 * det;
        if disc < 0.0 {
            return Err("Complex eigenvalues not supported".to_string());
        }
        let sqrt_disc = disc.sqrt();
        let lambda1 = (tr + sqrt_disc) / 2.0;
        let lambda2 = (tr - sqrt_disc) / 2.0;
        let eigvals = Array2::from_shape_vec((2, 1), vec![lambda1, lambda2]).unwrap();
        let mut eigvecs = Array2::zeros((2, 2));
        if a10.abs() > 1e-14 {
            let v0 = lambda1 - a11;
            let norm = (v0 * v0 + a10 * a10).sqrt();
            eigvecs[[0, 0]] = v0 / norm;
            eigvecs[[1, 0]] = a10 / norm;
            let v0 = lambda2 - a11;
            let norm = (v0 * v0 + a10 * a10).sqrt();
            eigvecs[[0, 1]] = v0 / norm;
            eigvecs[[1, 1]] = a10 / norm;
        } else if a01.abs() > 1e-14 {
            let v1 = lambda1 - a00;
            let norm = (a01 * a01 + v1 * v1).sqrt();
            eigvecs[[0, 0]] = a01 / norm;
            eigvecs[[1, 0]] = v1 / norm;
            let v1 = lambda2 - a00;
            let norm = (a01 * a01 + v1 * v1).sqrt();
            eigvecs[[0, 1]] = a01 / norm;
            eigvecs[[1, 1]] = v1 / norm;
        } else {
            eigvecs[[0, 0]] = 1.0;
            eigvecs[[1, 1]] = 1.0;
        }
        return Ok((eigvals, eigvecs));
    }
    let mat = ndarray_to_nalgebra(a);
    let eig = mat.symmetric_eigen();
    let eigvals = Array2::from_shape_vec((n, 1), eig.eigenvalues.iter().cloned().collect()).unwrap();
    let eigvecs = nalgebra_to_ndarray(eig.eigenvectors);
    Ok((eigvals, eigvecs))
}

/// Eigenvalue decomposition of a symmetric matrix.
/// Returns (sorted eigenvalues, eigenvectors as columns).
pub fn eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), String> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err("Matrix must be square".to_string());
    }
    let n = nrows;
    if n == 0 {
        return Err("Matrix must be non-empty".to_string());
    }
    let mat = ndarray_to_nalgebra(a);
    let eig = mat.symmetric_eigen();
    let mut pairs: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n);
    for i in 0..n {
        let val = eig.eigenvalues[i];
        let mut vec_col = Vec::with_capacity(n);
        for j in 0..n {
            vec_col.push(eig.eigenvectors[(j, i)]);
        }
        pairs.push((val, vec_col));
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let eigvals = Array1::from_vec(pairs.iter().map(|p| p.0).collect());
    let mut eigvecs = Array2::zeros((n, n));
    for (i, (_, vec_col)) in pairs.iter().enumerate() {
        for (j, &v) in vec_col.iter().enumerate() {
            eigvecs[[j, i]] = v;
        }
    }
    Ok((eigvals, eigvecs))
}

/// Least-squares solution via SVD.
/// Returns (solution, residual norm, rank, singular values).
pub fn lstsq(
    a: &Array2<f64>,
    b: &Array1<f64>,
    rcond: Option<f64>,
) -> Result<(Array1<f64>, f64, usize, Array1<f64>), String> {
    let (m, n) = a.dim();
    if b.len() != m {
        return Err("Dimension mismatch".to_string());
    }
    let svd_result = svd(a);
    let s = &svd_result.s;
    let u = svd_result.u.as_ref().ok_or("SVD U missing")?;
    let vt = svd_result.vh.as_ref().ok_or("SVD Vh missing")?;
    let max_s = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = rcond.unwrap_or(f64::EPSILON * n.max(m) as f64) * max_s;
    let rank = s.iter().filter(|&&si| si > threshold).count();
    let mut x = Array1::zeros(n);
    let ut_b = u.t().dot(b);
    for i in 0..rank {
        let coeff = ut_b[i] / s[i];
        for j in 0..n {
            x[j] += coeff * vt[[i, j]];
        }
    }
    let residual = b.to_owned() - a.dot(&x);
    let res_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok((x, res_norm, rank, s.to_owned()))
}

/// QR decomposition using Householder reflections.
/// Returns (Q, R).
pub fn qr(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let mat = ndarray_to_nalgebra(a);
    let qr_decomp = mat.qr();
    let q = nalgebra_to_ndarray(qr_decomp.q());
    let r = nalgebra_to_ndarray(qr_decomp.unpack_r());
    Ok((q, r))
}

/// LU decomposition with partial pivoting.
/// Returns (P, L, U).
pub fn lu(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), String> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err("Matrix must be square".to_string());
    }
    let n = nrows;
    let (lu_mat, piv) = lu_factor(a)?;
    let mut p = Array2::zeros((n, n));
    for i in 0..n {
        p[[i, piv[i]]] = 1.0;
    }
    let mut l = Array2::zeros((n, n));
    let mut u = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if j < i {
                l[[i, j]] = lu_mat[[i, j]];
            } else if i == j {
                l[[i, j]] = 1.0;
                u[[i, j]] = lu_mat[[i, j]];
            } else {
                u[[i, j]] = lu_mat[[i, j]];
            }
        }
    }
    Ok((p, l, u))
}

/// Compact LU factorization.
/// Returns (LU matrix, pivot indices).
pub fn lu_factor(a: &Array2<f64>) -> Result<(Array2<f64>, Vec<usize>), String> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err("Matrix must be square".to_string());
    }
    let n = nrows;
    let mut lu = a.to_owned();
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut max_val = lu[[k, k]].abs();
        let mut max_idx = k;
        for i in (k + 1)..n {
            if lu[[i, k]].abs() > max_val {
                max_val = lu[[i, k]].abs();
                max_idx = i;
            }
        }
        if max_val < 1e-14 {
            return Err("Matrix is singular".to_string());
        }
        if max_idx != k {
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_idx, j]];
                lu[[max_idx, j]] = tmp;
            }
            piv.swap(k, max_idx);
        }
        for i in (k + 1)..n {
            lu[[i, k]] /= lu[[k, k]];
            for j in (k + 1)..n {
                lu[[i, j]] -= lu[[i, k]] * lu[[k, j]];
            }
        }
    }
    Ok((lu, piv))
}

/// Solve a linear system using a pre-computed LU factorization.
pub fn lu_solve(lu: &Array2<f64>, piv: &[usize], b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n = lu.dim().0;
    if b.len() != n {
        return Err("Dimension mismatch".to_string());
    }
    let mut x = Array1::zeros(n);
    for i in 0..n {
        x[i] = b[piv[i]];
    }
    for i in 0..n {
        for j in 0..i {
            x[i] -= lu[[i, j]] * x[j];
        }
    }
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= lu[[i, j]] * x[j];
        }
        x[i] /= lu[[i, i]];
    }
    Ok(x)
}

/// Moore-Penrose pseudo-inverse via SVD.
pub fn pinv(a: &Array2<f64>, rcond: f64) -> Array2<f64> {
    let svd_result = svd(a);
    let s = &svd_result.s;
    let u = svd_result.u.as_ref().unwrap();
    let vt = svd_result.vh.as_ref().unwrap();
    let max_s = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = rcond * max_s;
    let (m, _n) = a.dim();
    let (_k, n_cols) = vt.dim();
    let mut s_inv = Array2::zeros((n_cols, m));
    for i in 0..s.len() {
        if s[i] > threshold {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }
    vt.t().dot(&s_inv).dot(&u.t())
}

/// Matrix and vector norms.
/// Supported: "1", "2", "inf", "fro" (Frobenius), "max".
pub fn norm(a: &Array2<f64>, ord: &str) -> f64 {
    let (m, n) = a.dim();
    match ord {
        "fro" => a.iter().map(|v| v * v).sum::<f64>().sqrt(),
        "max" => a.iter().cloned().fold(0.0_f64, |acc, v| acc.max(v.abs())),
        "1" => {
            let mut max_sum: f64 = 0.0;
            for j in 0..n {
                let col_sum: f64 = (0..m).map(|i| a[[i, j]].abs()).sum();
                max_sum = max_sum.max(col_sum);
            }
            max_sum
        }
        "inf" => {
            let mut max_sum: f64 = 0.0;
            for i in 0..m {
                let row_sum: f64 = (0..n).map(|j| a[[i, j]].abs()).sum();
                max_sum = max_sum.max(row_sum);
            }
            max_sum
        }
        "2" => {
            let s = svdvals(a);
            s.iter().cloned().fold(0.0_f64, f64::max)
        }
        _ => panic!("Unsupported norm order: {}", ord),
    }
}

/// Matrix rank via SVD.
pub fn matrix_rank(a: &Array2<f64>, tol: Option<f64>) -> usize {
    let s = svdvals(a);
    let (m, n) = a.dim();
    let max_s = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = tol.unwrap_or(f64::EPSILON * n.max(m) as f64) * max_s;
    s.iter().filter(|&&si| si > threshold).count()
}

/// Kronecker product of two matrices.
pub fn kron(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();
    let mut result = Array2::zeros((m * p, n * q));
    for i in 0..m {
        for j in 0..n {
            let aij = a[[i, j]];
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = aij * b[[k, l]];
                }
            }
        }
    }
    result
}

/// Block diagonal matrix from a slice of matrices.
pub fn block_diag(arrays: &[Array2<f64>]) -> Array2<f64> {
    let mut total_rows = 0;
    let mut total_cols = 0;
    for arr in arrays {
        let (r, c) = arr.dim();
        total_rows += r;
        total_cols += c;
    }
    let mut result = Array2::zeros((total_rows, total_cols));
    let mut row_offset = 0;
    let mut col_offset = 0;
    for arr in arrays {
        let (r, c) = arr.dim();
        for i in 0..r {
            for j in 0..c {
                result[[row_offset + i, col_offset + j]] = arr[[i, j]];
            }
        }
        row_offset += r;
        col_offset += c;
    }
    result
}

/// Orthonormal basis for the null space of a matrix via SVD.
pub fn null_space(a: &Array2<f64>, rcond: f64) -> Array2<f64> {
    let svd_result = svd(a);
    let s = &svd_result.s;
    let vt = svd_result.vh.as_ref().unwrap();
    let max_s = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = rcond * max_s;
    let (_m, n) = a.dim();
    let rank = s.iter().filter(|&&si| si > threshold).count();
    let ns_cols = n - rank;
    if ns_cols == 0 {
        return Array2::zeros((n, 0));
    }
    let mut ns = Array2::zeros((n, ns_cols));
    for j in 0..ns_cols {
        for i in 0..n {
            ns[[i, j]] = vt[[rank + j, i]];
        }
    }
    ns
}

/// Orthonormal basis for the range (column space) of a matrix via SVD.
pub fn orth(a: &Array2<f64>, rcond: f64) -> Array2<f64> {
    let svd_result = svd(a);
    let s = &svd_result.s;
    let u = svd_result.u.as_ref().unwrap();
    let max_s = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = rcond * max_s;
    let (m, _n) = a.dim();
    let rank = s.iter().filter(|&&si| si > threshold).count();
    if rank == 0 {
        return Array2::zeros((m, 0));
    }
    let mut o = Array2::zeros((m, rank));
    for j in 0..rank {
        for i in 0..m {
            o[[i, j]] = u[[i, j]];
        }
    }
    o
}

/// Matrix exponential via Padé approximant with scaling and squaring.
pub fn expm(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err("Matrix must be square".to_string());
    }
    let n = nrows;
    if n == 0 {
        return Err("Matrix must be non-empty".to_string());
    }
    let norm_a = norm(a, "1");
    let s = ((norm_a / 0.1).log2().ceil().max(0.0)) as usize;
    let mut a_scaled = a.to_owned();
    if s > 0 {
        let scale = 1.0 / (1_usize << s) as f64;
        for v in a_scaled.iter_mut() {
            *v *= scale;
        }
    }
    let a2 = a_scaled.dot(&a_scaled);
    let a3 = a2.dot(&a_scaled);
    let mut eye = Array2::zeros((n, n));
    for i in 0..n {
        eye[[i, i]] = 1.0;
    }
    let n_coeff = &eye + &a_scaled * 0.5 + &a2 * (1.0 / 10.0) + &a3 * (1.0 / 120.0);
    let d_coeff = &eye - &a_scaled * 0.5 + &a2 * (1.0 / 10.0) - &a3 * (1.0 / 120.0);
    let x = solve(&d_coeff, &n_coeff)?;
    let mut result = x;
    for _ in 0..s {
        result = result.dot(&result);
    }
    Ok(result)
}

/// Solve a triangular system via back/forward substitution.
pub fn solve_triangular(
    a: &Array2<f64>,
    b: &Array1<f64>,
    lower: bool,
) -> Result<Array1<f64>, String> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err("Matrix must be square".to_string());
    }
    if b.len() != nrows {
        return Err("Dimension mismatch".to_string());
    }
    let n = nrows;
    let mut x = Array1::zeros(n);
    if lower {
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= a[[i, j]] * x[j];
            }
            if a[[i, i]].abs() < 1e-14 {
                return Err("Matrix is singular".to_string());
            }
            x[i] = sum / a[[i, i]];
        }
    } else {
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= a[[i, j]] * x[j];
            }
            if a[[i, i]].abs() < 1e-14 {
                return Err("Matrix is singular".to_string());
            }
            x[i] = sum / a[[i, i]];
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[allow(dead_code)]
    fn approx_eq_arr1(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    fn approx_eq_arr2(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> bool {
        a.dim() == b.dim()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| approx_eq(*x, *y, tol))
    }

    #[test]
    fn test_inv() {
        let a = arr2(&[[2.0, 1.0], [5.0, 3.0]]);
        let a_inv = inv(&a).unwrap();
        let product = a.dot(&a_inv);
        let eye = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert!(approx_eq_arr2(&product, &eye, 1e-10));
    }

    #[test]
    fn test_det() {
        let a = arr2(&[[2.0, 1.0], [5.0, 3.0]]);
        assert!(approx_eq(det(&a), 1.0, 1e-10));
    }

    #[test]
    fn test_solve() {
        let a = arr2(&[[3.0, 1.0], [1.0, 2.0]]);
        let b = arr2(&[[9.0], [8.0]]);
        let x = solve(&a, &b).unwrap();
        assert!(approx_eq(x[[0, 0]], 2.0, 1e-10));
        assert!(approx_eq(x[[1, 0]], 3.0, 1e-10));
    }

    #[test]
    fn test_solve_vec() {
        let a = arr2(&[[3.0, 1.0], [1.0, 2.0]]);
        let b = arr1(&[9.0, 8.0]);
        let x = solve_vec(&a, &b).unwrap();
        assert!(approx_eq(x[0], 2.0, 1e-10));
        assert!(approx_eq(x[1], 3.0, 1e-10));
    }

    #[test]
    fn test_cholesky() {
        let a = arr2(&[[4.0, 2.0], [2.0, 3.0]]);
        let l = cholesky(&a, true).unwrap();
        let reconstructed = l.dot(&l.t());
        assert!(approx_eq_arr2(&reconstructed, &a, 1e-10));
    }

    #[test]
    fn test_svd() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let result = svd(&a);
        let u = result.u.unwrap();
        let s = result.s;
        let vh = result.vh.unwrap();
        let m = u.dot(&Array2::from_diag(&s).dot(&vh));
        assert!(approx_eq_arr2(&m, &a, 1e-10));
    }

    #[test]
    fn test_svdvals() {
        let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
        let s = svdvals(&a);
        assert!(approx_eq(s[0], 2.0, 1e-10));
        assert!(approx_eq(s[1], 1.0, 1e-10));
    }

    #[test]
    fn test_eig_identity() {
        let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let (eigvals, _eigvecs) = eig(&a).unwrap();
        assert!(approx_eq(eigvals[[0, 0]], 1.0, 1e-10));
        assert!(approx_eq(eigvals[[1, 0]], 1.0, 1e-10));
    }

    #[test]
    fn test_eig_diagonal() {
        let a = arr2(&[[3.0, 0.0], [0.0, 5.0]]);
        let (eigvals, eigvecs) = eig(&a).unwrap();
        let mut vals = vec![eigvals[[0, 0]], eigvals[[1, 0]]];
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(vals[0], 3.0, 1e-10));
        assert!(approx_eq(vals[1], 5.0, 1e-10));
        let a_recon = eigvecs.dot(&Array2::from_diag(&Array1::from_vec(vals)).dot(&eigvecs.t()));
        assert!(approx_eq_arr2(&a_recon, &a, 1e-8));
    }

    #[test]
    fn test_eigh_symmetric() {
        let a = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let (eigvals, eigvecs) = eigh(&a).unwrap();
        assert!(approx_eq(eigvals[0], 1.0, 1e-10));
        assert!(approx_eq(eigvals[1], 3.0, 1e-10));
        let a_recon = eigvecs.dot(&Array2::from_diag(&eigvals).dot(&eigvecs.t()));
        assert!(approx_eq_arr2(&a_recon, &a, 1e-8));
    }

    #[test]
    fn test_lstsq() {
        let a = arr2(&[[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]);
        let b = arr1(&[1.0, 2.0, 2.0]);
        let (x, _res, rank, _s) = lstsq(&a, &b, None).unwrap();
        assert_eq!(rank, 2);
        assert!(approx_eq(x[0], 0.666666666666, 1e-6));
        assert!(approx_eq(x[1], 0.5, 1e-6));
    }

    #[test]
    fn test_qr() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let (q, r) = qr(&a).unwrap();
        let recon = q.dot(&r);
        assert!(approx_eq_arr2(&recon, &a, 1e-10));
        let qtq = q.t().dot(&q);
        let eye = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert!(approx_eq_arr2(&qtq, &eye, 1e-10));
    }

    #[test]
    fn test_lu() {
        let a = arr2(&[[2.0, 1.0], [5.0, 3.0]]);
        let (p, l, u) = lu(&a).unwrap();
        let recon = p.dot(&l.dot(&u));
        assert!(approx_eq_arr2(&recon, &a, 1e-10));
    }

    #[test]
    fn test_lu_factor_and_solve() {
        let a = arr2(&[[3.0, 1.0], [1.0, 2.0]]);
        let b = arr1(&[9.0, 8.0]);
        let (lu_mat, piv) = lu_factor(&a).unwrap();
        let x = lu_solve(&lu_mat, &piv, &b).unwrap();
        assert!(approx_eq(x[0], 2.0, 1e-10));
        assert!(approx_eq(x[1], 3.0, 1e-10));
    }

    #[test]
    fn test_pinv() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let ai = pinv(&a, f64::EPSILON);
        let product = a.dot(&ai);
        let eye = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert!(approx_eq_arr2(&product, &eye, 1e-10));
    }

    #[test]
    fn test_norm_frobenius() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(approx_eq(norm(&a, "fro"), (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt(), 1e-10));
    }

    #[test]
    fn test_norm_1() {
        let a = arr2(&[[1.0, -2.0], [3.0, 4.0]]);
        assert!(approx_eq(norm(&a, "1"), 6.0, 1e-10));
    }

    #[test]
    fn test_norm_inf() {
        let a = arr2(&[[1.0, -2.0], [3.0, 4.0]]);
        assert!(approx_eq(norm(&a, "inf"), 7.0, 1e-10));
    }

    #[test]
    fn test_norm_max() {
        let a = arr2(&[[1.0, -5.0], [3.0, 4.0]]);
        assert!(approx_eq(norm(&a, "max"), 5.0, 1e-10));
    }

    #[test]
    fn test_norm_2() {
        let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
        assert!(approx_eq(norm(&a, "2"), 2.0, 1e-10));
    }

    #[test]
    fn test_matrix_rank() {
        let a = arr2(&[[1.0, 2.0], [2.0, 4.0]]);
        assert_eq!(matrix_rank(&a, None), 1);
        let b = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(matrix_rank(&b, None), 2);
    }

    #[test]
    fn test_kron() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[0.0, 5.0], [6.0, 7.0]]);
        let expected = arr2(&[
            [0.0, 5.0, 0.0, 10.0],
            [6.0, 7.0, 12.0, 14.0],
            [0.0, 15.0, 0.0, 20.0],
            [18.0, 21.0, 24.0, 28.0],
        ]);
        assert!(approx_eq_arr2(&kron(&a, &b), &expected, 1e-10));
    }

    #[test]
    fn test_block_diag() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0]]);
        let c = block_diag(&[a, b]);
        let expected = arr2(&[[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]);
        assert!(approx_eq_arr2(&c, &expected, 1e-10));
    }

    #[test]
    fn test_null_space() {
        let a = arr2(&[[1.0, 2.0], [2.0, 4.0]]);
        let ns = null_space(&a, f64::EPSILON);
        assert_eq!(ns.dim().1, 1);
        let prod = a.dot(&ns);
        assert!(approx_eq(prod[[0, 0]], 0.0, 1e-10));
        assert!(approx_eq(prod[[1, 0]], 0.0, 1e-10));
    }

    #[test]
    fn test_orth() {
        let a = arr2(&[[1.0, 2.0], [2.0, 4.0]]);
        let o = orth(&a, f64::EPSILON);
        assert_eq!(o.dim().1, 1);
        let dot = o[[0, 0]] * o[[0, 0]] + o[[1, 0]] * o[[1, 0]];
        assert!(approx_eq(dot, 1.0, 1e-10));
    }

    #[test]
    fn test_expm_zero() {
        let a = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
        let result = expm(&a).unwrap();
        let eye = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert!(approx_eq_arr2(&result, &eye, 1e-10));
    }

    #[test]
    fn test_expm_diagonal() {
        let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
        let result = expm(&a).unwrap();
        assert!(approx_eq(result[[0, 0]], 1.0_f64.exp(), 1e-6));
        assert!(approx_eq(result[[1, 1]], 2.0_f64.exp(), 1e-6));
        assert!(approx_eq(result[[0, 1]], 0.0, 1e-10));
        assert!(approx_eq(result[[1, 0]], 0.0, 1e-10));
    }

    #[test]
    fn test_solve_triangular_lower() {
        let l = arr2(&[[2.0, 0.0], [1.0, 3.0]]);
        let b = arr1(&[4.0, 7.0]);
        let x = solve_triangular(&l, &b, true).unwrap();
        assert!(approx_eq(x[0], 2.0, 1e-10));
        assert!(approx_eq(x[1], 5.0 / 3.0, 1e-10));
    }

    #[test]
    fn test_solve_triangular_upper() {
        let u = arr2(&[[2.0, 1.0], [0.0, 3.0]]);
        let b = arr1(&[5.0, 9.0]);
        let x = solve_triangular(&u, &b, false).unwrap();
        assert!(approx_eq(x[0], 1.0, 1e-10));
        assert!(approx_eq(x[1], 3.0, 1e-10));
    }
}
