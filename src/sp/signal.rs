#![allow(dead_code)]

use ndarray::Array1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConvolveMode {
    Full,
    Same,
    Valid,
}

/// 1D Convolution of two signals.
pub fn convolve(in1: &Array1<f64>, in2: &Array1<f64>, mode: ConvolveMode) -> Array1<f64> {
    let n1 = in1.len();
    let n2 = in2.len();
    if n1 == 0 || n2 == 0 {
        return Array1::from_vec(vec![]);
    }

    let out_len = n1 + n2 - 1;
    let mut full_result = vec![0.0; out_len];

    for i in 0..n1 {
        for j in 0..n2 {
            full_result[i + j] += in1[i] * in2[j];
        }
    }

    match mode {
        ConvolveMode::Full => Array1::from_vec(full_result),
        ConvolveMode::Same => {
            let start = (n2 - 1) / 2;
            let end = start + n1;
            Array1::from_vec(full_result[start..end].to_vec())
        }
        ConvolveMode::Valid => {
            if n1 >= n2 {
                let start = n2 - 1;
                let end = n1;
                Array1::from_vec(full_result[start..end].to_vec())
            } else {
                let start = n1 - 1;
                let end = n2;
                Array1::from_vec(full_result[start..end].to_vec())
            }
        }
    }
}

/// Filter data along one dimension with an IIR or FIR filter using Transposed Direct Form II.
pub fn lfilter(b: &Array1<f64>, a: &Array1<f64>, x: &Array1<f64>) -> Result<Array1<f64>, String> {
    if a.len() == 0 || b.len() == 0 {
        return Err("Filter coefficients a and b must not be empty".to_string());
    }
    let a0 = a[0];
    if a0.abs() < 1e-14 {
        return Err("First denominator coefficient a[0] must not be zero".to_string());
    }

    let b_norm: Vec<f64> = b.iter().map(|&x| x / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&x| x / a0).collect();

    let n = x.len();
    let mut y = Array1::zeros(n);

    let n_b = b_norm.len();
    let n_a = a_norm.len();
    let ord = std::cmp::max(n_b, n_a);
    let mut z = vec![0.0; ord + 1];

    for i in 0..n {
        let xi = x[i];
        let yi = b_norm[0] * xi + z[0];
        y[i] = yi;

        for k in 1..ord {
            let bk = if k < n_b { b_norm[k] } else { 0.0 };
            let ak = if k < n_a { a_norm[k] } else { 0.0 };
            z[k - 1] = bk * xi - ak * yi + z[k];
        }
    }

    Ok(y)
}

/// Zero-phase forward-backward digital filter.
pub fn filtfilt(b: &Array1<f64>, a: &Array1<f64>, x: &Array1<f64>) -> Result<Array1<f64>, String> {
    let padlen = 3 * std::cmp::max(b.len(), a.len());
    if x.len() <= padlen {
        return Err("Input signal is too short for the requested padlen".to_string());
    }

    // Construct padded signal with reflection padding
    let mut padded_vec = Vec::with_capacity(x.len() + 2 * padlen);

    let x0 = x[0];
    for i in (1..=padlen).rev() {
        padded_vec.push(2.0 * x0 - x[i]);
    }

    for &val in x.iter() {
        padded_vec.push(val);
    }

    let xn = x[x.len() - 1];
    let n = x.len();
    for i in 1..=padlen {
        padded_vec.push(2.0 * xn - x[n - 1 - i]);
    }

    let padded = Array1::from_vec(padded_vec);

    // Forward filter
    let y1 = lfilter(b, a, &padded)?;

    // Reverse y1
    let mut y1_rev = y1.to_vec();
    y1_rev.reverse();
    let y1_rev_arr = Array1::from_vec(y1_rev);

    // Backward filter
    let y2 = lfilter(b, a, &y1_rev_arr)?;

    // Reverse y2 back
    let mut y2_rev = y2.to_vec();
    y2_rev.reverse();

    // Crop the padded boundaries
    let start_idx = padlen;
    let end_idx = padlen + x.len();
    let cropped = y2_rev[start_idx..end_idx].to_vec();

    Ok(Array1::from_vec(cropped))
}

/// Find local peaks (maxima) in a 1D signal.
pub fn find_peaks(
    x: &Array1<f64>,
    height: Option<f64>,
    distance: Option<usize>,
) -> Vec<usize> {
    let n = x.len();
    if n < 3 {
        return vec![];
    }

    let mut peaks = Vec::new();
    for i in 1..(n - 1) {
        if x[i] > x[i - 1] && x[i] > x[i + 1] {
            if let Some(h) = height {
                if x[i] < h {
                    continue;
                }
            }
            peaks.push(i);
        }
    }

    if let Some(dist) = distance {
        let mut kept = Vec::new();
        // Sort peaks by height descending
        let mut peak_heights: Vec<(usize, f64)> = peaks.iter().map(|&idx| (idx, x[idx])).collect();
        peak_heights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, _h) in peak_heights {
            let mut ok = true;
            for &k_idx in &kept {
                let diff = if idx > k_idx { idx - k_idx } else { k_idx - idx };
                if diff < dist {
                    ok = false;
                    break;
                }
            }
            if ok {
                kept.push(idx);
            }
        }
        kept.sort();
        peaks = kept;
    }

    peaks
}

// ── Helpers for polynomial and filter design ───────────────────────────────

fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            c[i + j] += ai * bj;
        }
    }
    c
}

fn poly_eval(p: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for &c in p {
        result = result * x + c;
    }
    result
}

fn polyder(p: &[f64]) -> Vec<f64> {
    let n = p.len();
    if n <= 1 {
        return vec![0.0];
    }
    let mut d = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        d.push(p[i] * (n - 1 - i) as f64);
    }
    d
}

fn poly_from_roots(roots: &[(f64, f64)]) -> Vec<f64> {
    let n = roots.len();
    let mut result_re = vec![0.0; n + 1];
    let mut result_im = vec![0.0; n + 1];
    result_re[0] = 1.0;

    for &(re, im) in roots {
        for j in (1..=n).rev() {
            let rj_re = result_re[j];
            let rj_im = result_im[j];
            let rj1_re = result_re[j - 1];
            let rj1_im = result_im[j - 1];
            let prod_re = re * rj_re - im * rj_im;
            let prod_im = re * rj_im + im * rj_re;
            result_re[j] = rj1_re - prod_re;
            result_im[j] = rj1_im - prod_im;
        }
    }
    result_re
}

/// Solve lower-triangular system Lx = b via forward substitution.
fn solve_lower_triangular(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = s / l[i][i];
    }
    x
}

fn solve_upper_triangular(u: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= u[i][j] * x[j];
        }
        x[i] = s / u[i][i];
    }
    x
}

/// Solve Ax = b via LU decomposition with partial pivoting.
fn solve_linear_system(a_in: &[Vec<f64>], b_in: &[f64]) -> Vec<f64> {
    let n = b_in.len();
    let mut a: Vec<Vec<f64>> = a_in.iter().map(|row| row.clone()).collect();
    let mut b: Vec<f64> = b_in.to_vec();

    for k in 0..n {
        let mut max_val = a[k][k].abs();
        let mut max_row = k;
        for i in k + 1..n {
            if a[i][k].abs() > max_val {
                max_val = a[i][k].abs();
                max_row = i;
            }
        }
        if max_row != k {
            a.swap(k, max_row);
            b.swap(k, max_row);
        }
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k + 1..n {
                a[i][j] -= factor * a[k][j];
            }
            a[i][k] = factor;
        }
    }

    // Forward substitution (L)
    for i in 1..n {
        for j in 0..i {
            b[i] -= a[i][j] * b[j];
        }
    }

    // Back substitution (U)
    for i in (0..n).rev() {
        for j in i + 1..n {
            b[i] -= a[i][j] * b[j];
        }
        b[i] /= a[i][i];
    }

    b
}

/// Compute Q matrix from Householder reflections.
fn compute_q_matrix(q_accum: &[Vec<(f64, f64)>], n: usize) -> Vec<Vec<(f64, f64)>> {
    let mut q = vec![vec![(0.0, 0.0); n]; n];
    for i in 0..n {
        q[i][i] = (1.0, 0.0);
    }
    for k in (0..n).rev() {
        let mut new_q = vec![vec![(0.0, 0.0); n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = (0.0, 0.0);
                for l in 0..n {
                    let a = q_accum[k][l];
                    let b = q[l][j];
                    s.0 += a.0 * b.0 - a.1 * b.1;
                    s.1 += a.0 * b.1 + a.1 * b.0;
                }
                new_q[i][j] = s;
            }
        }
        q = new_q;
    }
    q
}

/// Eigenvalues of upper Hessenberg matrix via QR iteration.
fn eigenvalues_hessenberg(mut h: Vec<Vec<(f64, f64)>>) -> Vec<(f64, f64)> {
    let n = h.len();
    if n == 0 {
        return vec![];
    }
    let mut q_accum: Vec<Vec<(f64, f64)>> = vec![vec![(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            q_accum[i][j] = if i == j { (1.0, 0.0) } else { (0.0, 0.0) };
        }
    }

    for _iter in 0..100 * n {
        if n == 1 {
            break;
        }
        // Check for convergence
        let mut all_converged = true;
        for i in 1..n {
            let mag = (h[i][i - 1].0 * h[i][i - 1].0 + h[i][i - 1].1 * h[i][i - 1].1).sqrt();
            if mag > 1e-14 {
                all_converged = false;
                break;
            }
        }
        if all_converged {
            break;
        }

        // Wilkinson shift
        let a = h[n - 2][n - 2];
        let b_val = h[n - 2][n - 1];
        let c = h[n - 1][n - 2];
        let d = h[n - 1][n - 1];

        let tr = (a.0 + d.0, a.1 + d.1);
        let det = (a.0 * d.0 - a.1 * d.1 - b_val.0 * c.0 + b_val.1 * c.1,
                    a.0 * d.1 + a.1 * d.0 - b_val.0 * c.1 - b_val.1 * c.0);
        let disc_sqrt = ((tr.0 * tr.0 - tr.1 * tr.1 - 4.0 * det.0).powi(2)
            + (2.0 * tr.0 * tr.1 - 4.0 * det.1).powi(2))
        .sqrt()
        .sqrt();
        let disc_angle = (2.0 * tr.0 * tr.1 - 4.0 * det.1).atan2(tr.0 * tr.0 - tr.1 * tr.1 - 4.0 * det.0) / 2.0;
        let disc = (disc_sqrt * disc_angle.cos(), disc_sqrt * disc_angle.sin());

        let mu1 = ((tr.0 + disc.0) / 2.0, (tr.1 + disc.1) / 2.0);
        let mu2 = ((tr.0 - disc.0) / 2.0, (tr.1 - disc.1) / 2.0);

        // Pick the shift closer to d
        let dist1 = ((d.0 - mu1.0).powi(2) + (d.1 - mu1.1).powi(2)).sqrt();
        let dist2 = ((d.0 - mu2.0).powi(2) + (d.1 - mu2.1).powi(2)).sqrt();
        let shift = if dist1 <= dist2 { mu1 } else { mu2 };

        // Apply shift
        for i in 0..n {
            h[i][i].0 -= shift.0;
            h[i][i].1 -= shift.1;
        }

        // QR decomposition via Givens rotations
        for i in 0..n - 1 {
            let a = h[i][i];
            let b = h[i + 1][i];
            let r = (a.0 * a.0 + a.1 * a.1 + b.0 * b.0 + b.1 * b.1).sqrt();
            if r < 1e-300 {
                continue;
            }
            let c = (a.0 / r, a.1 / r);
            let s = (b.0 / r, b.1 / r);

            for j in 0..n {
                let t1 = h[i][j];
                let t2 = h[i + 1][j];
                h[i][j] = (c.0 * t1.0 + c.1 * t1.1 + s.0 * t2.0 + s.1 * t2.1,
                           c.0 * t1.1 - c.1 * t1.0 + s.0 * t2.1 - s.1 * t2.0);
                h[i + 1][j] = (-s.0 * t1.0 + s.1 * t1.1 + c.0 * t2.0 - c.1 * t2.1,
                               -s.0 * t1.1 - s.1 * t1.0 + c.0 * t2.1 + c.1 * t2.0);
            }
            for j in 0..n {
                let hji = h[j][i];
                let hji1 = h[j][i + 1];
                h[j][i] = (c.0 * hji.0 - c.1 * hji.1 + s.0 * hji1.0 - s.1 * hji1.1,
                           c.0 * hji.1 + c.1 * hji.0 + s.0 * hji1.1 + s.1 * hji1.0);
                h[j][i + 1] = (-s.0 * hji.0 + s.1 * hji.1 + c.0 * hji1.0 - c.1 * hji1.1,
                               -s.0 * hji.1 - s.1 * hji.0 + c.0 * hji1.1 + c.1 * hji1.0);
            }
            let q1 = q_accum[i].clone();
            let q2 = q_accum[i + 1].clone();
            for j in 0..n {
                q_accum[i][j] = (c.0 * q1[j].0 - c.1 * q1[j].1 + s.0 * q2[j].0 - s.1 * q2[j].1,
                                  c.0 * q1[j].1 + c.1 * q1[j].0 + s.0 * q2[j].1 + s.1 * q2[j].0);
                q_accum[i + 1][j] = (-s.0 * q1[j].0 + s.1 * q1[j].1 + c.0 * q2[j].0 - c.1 * q2[j].1,
                                      -s.0 * q1[j].1 - s.1 * q1[j].0 + c.0 * q2[j].1 + c.1 * q2[j].0);
            }
        }

        // Apply shift back
        for i in 0..n {
            h[i][i].0 += shift.0;
            h[i][i].1 += shift.1;
        }
    }

    let q = compute_q_matrix(&q_accum, n);

    // Compute RQ = R * Q
    let mut rq = vec![vec![(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = (0.0, 0.0);
            for k in i..n.min(n) {
                let a = h[i][k];
                let b = q[k][j];
                s.0 += a.0 * b.0 - a.1 * b.1;
                s.1 += a.0 * b.1 + a.1 * b.0;
            }
            rq[i][j] = s;
        }
    }

    (0..n).map(|i| rq[i][i]).collect()
}

/// Roots of a polynomial with real coefficients (companion matrix eigenvalues).
fn roots_from_poly(poly: &[f64]) -> Vec<(f64, f64)> {
    let n = poly.len();
    if n <= 1 {
        return vec![];
    }

    let degree = n - 1;
    let lead = poly[0];

    // Companion matrix (complex)
    let mut h = vec![vec![(0.0, 0.0); degree]; degree];
    for i in 0..degree {
        h[0][i] = (-poly[i + 1] / lead, 0.0);
    }
    for i in 1..degree {
        h[i][i - 1] = (1.0, 0.0);
    }

    // Balance the companion matrix
    for _iter in 0..10 {
        let mut changed = false;
        for i in 0..degree {
            let col_norm = (h[0][i].0 * h[0][i].0 + h[0][i].1 * h[0][i].1).sqrt();
            let row_norm = if i + 1 < degree {
                (h[i + 1][i].0 * h[i + 1][i].0 + h[i + 1][i].1 * h[i + 1][i].1).sqrt()
            } else {
                0.0
            };
            if col_norm < 1e-30 || row_norm < 1e-30 {
                continue;
            }
            let s = col_norm + row_norm;
            let mut f = 1.0;
            let mut g = col_norm / s;
            while g < 0.5 {
                f *= 2.0;
                g *= 2.0;
            }
            g = row_norm / s;
            while g >= 2.0 {
                f /= 2.0;
                g /= 2.0;
            }
            if col_norm + row_norm != f * col_norm {
                changed = true;
                for j in 0..degree {
                    h[j][i].0 *= f;
                    h[j][i].1 *= f;
                }
                for j in 0..degree {
                    h[i][j].0 /= f;
                    h[i][j].1 /= f;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Reduce to upper Hessenberg via Householder reflections
    for k in 0..degree - 2 {
        let mut s = 0.0;
        for i in k + 1..degree {
            s += h[i][k].0 * h[i][k].0 + h[i][k].1 * h[i][k].1;
        }
        s = s.sqrt();
        if s < 1e-300 {
            continue;
        }

        let mut alpha = (0.0, 0.0);
        if h[k + 1][k].0.abs() > 1e-300 || h[k + 1][k].1.abs() > 1e-300 {
            let mag = (h[k + 1][k].0 * h[k + 1][k].0 + h[k + 1][k].1 * h[k + 1][k].1).sqrt();
            alpha.0 = -s * h[k + 1][k].0 / mag;
            alpha.1 = -s * h[k + 1][k].1 / mag;
        } else {
            alpha.0 = -s;
        }
        let mut v = vec![(0.0, 0.0); degree];
        for i in k + 1..degree {
            v[i] = h[i][k];
        }
        v[k + 1].0 -= alpha.0;
        v[k + 1].1 -= alpha.1;

        let v_norm_sq: f64 = v.iter().map(|x| x.0 * x.0 + x.1 * x.1).sum();
        if v_norm_sq < 1e-300 {
            continue;
        }

        // H' = (I - 2vv'/v'v) H (I - 2vv'/v'v)
        // w = H'v / (v'v/2)
        let mut hv = vec![(0.0, 0.0); degree];
        for j in 0..degree {
            let mut s = (0.0, 0.0);
            for i in k + 1..degree {
                let a = h[i][j];
                let b = v[i];
                s.0 += a.0 * b.0 + a.1 * b.1;
                s.1 += a.0 * b.1 - a.1 * b.0;
            }
            hv[j] = s;
        }

        let mut vt_h = vec![(0.0, 0.0); degree];
        for i in 0..degree {
            let mut s = (0.0, 0.0);
            for j in k + 1..degree {
                let a = h[i][j];
                let b = v[j];
                s.0 += a.0 * b.0 - a.1 * b.1;
                s.1 += a.0 * b.1 + a.1 * b.0;
            }
            vt_h[i] = s;
        }

        let scale = 2.0 / v_norm_sq;
        for i in 0..degree {
            for j in 0..degree {
                let term_re = scale * (v[i].0 * hv[j].0 - v[i].1 * hv[j].1 + vt_h[i].0 * v[j].0 - vt_h[i].1 * v[j].1);
                let term_im = scale * (v[i].0 * hv[j].1 + v[i].1 * hv[j].0 + vt_h[i].0 * v[j].1 + vt_h[i].1 * v[j].0);
                h[i][j].0 -= term_re;
                h[i][j].1 -= term_im;
            }
        }
    }

    // Zero out near-zero subdiagonal
    for i in 1..degree {
        if h[i][i - 1].0.abs() < 1e-14 && h[i][i - 1].1.abs() < 1e-14 {
            h[i][i - 1] = (0.0, 0.0);
        }
    }

    eigenvalues_hessenberg(h)
}

fn roots_to_coeffs(roots: &[(f64, f64)]) -> Vec<f64> {
    let n = roots.len();
    let mut result = vec![0.0; n + 1];
    result[0] = 1.0;
    for &(re, im) in roots {
        for j in (1..=n).rev() {
            // Multiply by (x - root)
            let a = result[j];
            let b = result[j - 1];
            result[j] = b - re * a + im * 0.0; // im part cancels for conjugate pairs
            result[j - 1] = a;
        }
    }
    result
}

fn roots_to_coeffs_conjugate_pairs(roots: &[(f64, f64)]) -> Vec<f64> {
    let n = roots.len();
    let mut result = vec![0.0; n + 1];
    result[0] = 1.0;
    for &(re, im) in roots {
        for j in (1..=n).rev() {
            let a = result[j];
            let b = result[j - 1];
            result[j] = b - re * a;
            result[j - 1] = a;
        }
        if im != 0.0 {
            // This was a conjugate pair, already handled
        }
    }
    result
}

/// Savitzky-Golay polynomial smoothing coefficients via least-squares.
fn savgol_coeffs(window_length: usize, polyorder: usize) -> Vec<f64> {
    let half = window_length / 2;
    // Build Vandermonde matrix for polynomial fitting
    let mut a = vec![vec![0.0; polyorder + 1]; window_length];
    for i in 0..window_length {
        let x = (i as f64) - (half as f64);
        let mut val = 1.0;
        for j in 0..=polyorder {
            a[i][j] = val;
            val *= x;
        }
    }

    // Solve normal equations: (A'A)c = A'y
    let m = window_length;
    let n = polyorder + 1;
    let mut ata = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    // Target: polynomial evaluated at center (x=0)
    let mut aty = vec![0.0; n];
    for i in 0..n {
        aty[i] = a[half][i];
    }

    let coeffs = solve_linear_system(&ata, &aty);

    // Evaluate polynomial at each point in the window to get convolution coefficients
    let mut result = vec![0.0; window_length];
    for i in 0..window_length {
        let x = (i as f64) - (half as f64);
        let mut val = 0.0;
        let mut xpow = 1.0;
        for j in 0..=polyorder {
            val += coeffs[j] * xpow;
            xpow *= x;
        }
        result[i] = val;
    }
    result
}

// ── Filter design ─────────────────────────────────────────────────────────

/// Butterworth filter design.
/// Returns (b_coefficients, a_coefficients, gain).
/// `btype` = "low" or "high".
pub fn butter(n: usize, wn: f64, btype: &str) -> (Vec<f64>, Vec<f64>, f64) {
    // Analog Butterworth poles: s_k = exp(j * pi * (2*k + n + 1) / (2*n))
    let mut analog_poles = Vec::with_capacity(n);
    for k in 0..n {
        let angle = std::f64::consts::PI * (2.0 * k as f64 + n as f64 + 1.0) / (2.0 * n as f64);
        analog_poles.push((angle.cos(), angle.sin()));
    }

    // Bilinear transform pre-warping
    let fs = 2.0;
    let warped = (std::f64::consts::PI * wn / fs).tan();

    // Scale poles by warped frequency
    let mut z_poles = Vec::with_capacity(n);
    for &(re, im) in &analog_poles {
        let scaled_re = re * warped;
        let scaled_im = im * warped;
        // Bilinear transform: z = (1 + s/2) / (1 - s/2)
        let sr = scaled_re / 2.0;
        let si = scaled_im / 2.0;
        let num_re = 1.0 + sr;
        let num_im = si;
        let den_re = 1.0 - sr;
        let den_im = -si;
        let den_mag_sq = den_re * den_re + den_im * den_im;
        z_poles.push((
            (num_re * den_re + num_im * den_im) / den_mag_sq,
            (num_im * den_re - num_re * den_im) / den_mag_sq,
        ));
    }

    // Denominator polynomial from z-domain poles
    let den = poly_from_roots(&z_poles);

    if btype == "low" {
        // All zeros at z = -1
        let zeros = vec![(-1.0, 0.0); n];
        let num_unscaled = poly_from_roots(&zeros);
        // DC gain = 1
        let den_dc = poly_eval(&den, 1.0);
        let num_dc = 2.0f64.powi(n as i32);
        let gain = den_dc / num_dc;
        let b: Vec<f64> = num_unscaled.iter().map(|&c| c * gain).collect();
        (b, den, gain)
    } else {
        // Highpass: zeros at z = 1
        let zeros = vec![(1.0, 0.0); n];
        let num_unscaled = poly_from_roots(&zeros);
        // Gain at z = -1 (high frequency)
        let den_hf = poly_eval(&den, -1.0);
        let num_hf = poly_eval(&num_unscaled, -1.0);
        let gain = den_hf / num_hf;
        let b: Vec<f64> = num_unscaled.iter().map(|&c| c * gain).collect();
        (b, den, gain)
    }
}

/// Chebyshev Type I filter design.
/// `rp` = passband ripple in dB.
pub fn cheby1(n: usize, rp: f64, wn: f64, btype: &str) -> (Vec<f64>, Vec<f64>, f64) {
    let eps = (10.0f64.powf(rp / 10.0) - 1.0).sqrt();
    let a = (1.0 / eps + (1.0 / (eps * eps) + 1.0).sqrt()).ln() / n as f64;

    // Analog Chebyshev Type I poles
    let mut analog_poles = Vec::with_capacity(n);
    for k in 0..n {
        let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * n) as f64;
        let sinh_a = a.sinh();
        let cosh_a = a.cosh();
        analog_poles.push((-sinh_a * theta.sin(), cosh_a * theta.cos()));
    }

    // Bilinear transform pre-warping
    let fs = 2.0;
    let warped = (std::f64::consts::PI * wn / fs).tan();

    // Scale poles by warped frequency and apply bilinear transform
    let mut z_poles = Vec::with_capacity(n);
    for &(re, im) in &analog_poles {
        let scaled_re = re * warped;
        let scaled_im = im * warped;
        let sr = scaled_re / 2.0;
        let si = scaled_im / 2.0;
        let num_re = 1.0 + sr;
        let num_im = si;
        let den_re = 1.0 - sr;
        let den_im = -si;
        let den_mag_sq = den_re * den_re + den_im * den_im;
        z_poles.push((
            (num_re * den_re + num_im * den_im) / den_mag_sq,
            (num_im * den_re - num_re * den_im) / den_mag_sq,
        ));
    }

    let den = poly_from_roots(&z_poles);

    if btype == "low" {
        let zeros = vec![(-1.0, 0.0); n];
        let num_unscaled = poly_from_roots(&zeros);
        // DC gain for Chebyshev Type I: 1 / sqrt(1 + eps^2) for even n, 1 for odd n
        let gain_factor = if n % 2 == 0 {
            (1.0 + eps * eps).sqrt()
        } else {
            1.0
        };
        let den_dc = poly_eval(&den, 1.0);
        let num_dc = 2.0f64.powi(n as i32);
        let gain = den_dc / (num_dc * gain_factor);
        let b: Vec<f64> = num_unscaled.iter().map(|&c| c * gain).collect();
        (b, den, gain)
    } else {
        let zeros = vec![(1.0, 0.0); n];
        let num_unscaled = poly_from_roots(&zeros);
        let gain_factor = if n % 2 == 0 {
            (1.0 + eps * eps).sqrt()
        } else {
            1.0
        };
        let den_hf = poly_eval(&den, -1.0);
        let num_hf = poly_eval(&num_unscaled, -1.0);
        let gain = den_hf / (num_hf * gain_factor);
        let b: Vec<f64> = num_unscaled.iter().map(|&c| c * gain).collect();
        (b, den, gain)
    }
}

/// Filter signal x using second-order sections.
/// Each section is [b0, b1, b2, a0, a1, a2].
/// Apply sections sequentially using Direct Form II.
pub fn sosfilt(sos: &[[f64; 6]], x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for section in sos {
        let a0 = section[3];
        let b0 = section[0] / a0;
        let b1 = section[1] / a0;
        let b2 = section[2] / a0;
        let a1 = section[4] / a0;
        let a2 = section[5] / a0;
        let mut w1 = 0.0;
        let mut w2 = 0.0;
        for i in 0..y.len() {
            let xi = y[i];
            let w0 = xi - a1 * w1 - a2 * w2;
            y[i] = b0 * w0 + b1 * w1 + b2 * w2;
            w2 = w1;
            w1 = w0;
        }
    }
    y
}

/// Zero-phase filtering using second-order sections.
/// Forward filter, reverse, filter again, reverse.
pub fn sosfiltfilt(sos: &[[f64; 6]], x: &[f64]) -> Vec<f64> {
    let y1 = sosfilt(sos, x);
    let mut y1_rev = y1;
    y1_rev.reverse();
    let y2 = sosfilt(sos, &y1_rev);
    let mut result = y2;
    result.reverse();
    result
}

/// Frequency response of a digital filter.
/// Returns (frequencies, complex_response_magnitudes).
/// Evaluate H(e^{jw}) at N evenly spaced points in [0, pi).
pub fn freqz(b: &[f64], a: &[f64], wor_n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(wor_n);
    let mut mags = Vec::with_capacity(wor_n);
    for i in 0..wor_n {
        let w = std::f64::consts::PI * i as f64 / wor_n as f64;
        freqs.push(w);
        // Evaluate numerator and denominator at e^{jw}
        let mut num_re = 0.0;
        let mut num_im = 0.0;
        for (k, &bk) in b.iter().enumerate() {
            let angle = w * k as f64;
            num_re += bk * angle.cos();
            num_im -= bk * angle.sin();
        }
        let mut den_re = 0.0;
        let mut den_im = 0.0;
        for (k, &ak) in a.iter().enumerate() {
            let angle = w * k as f64;
            den_re += ak * angle.cos();
            den_im -= ak * angle.sin();
        }
        let den_mag_sq = den_re * den_re + den_im * den_im;
        if den_mag_sq > 1e-30 {
            let h_re = (num_re * den_re + num_im * den_im) / den_mag_sq;
            let h_im = (num_im * den_re - num_re * den_im) / den_mag_sq;
            mags.push((h_re * h_re + h_im * h_im).sqrt());
        } else {
            mags.push(0.0);
        }
    }
    (freqs, mags)
}

// ── Convolution/correlation ────────────────────────────────────────────────

/// Cross-correlation of two signals.
/// Modes: "full" (default), "valid", "same".
pub fn correlate(a: &[f64], b: &[f64], mode: &str) -> Vec<f64> {
    let mut b_rev = b.to_vec();
    b_rev.reverse();
    let a_arr = Array1::from_vec(a.to_vec());
    let b_arr = Array1::from_vec(b_rev);
    let conv_mode = match mode {
        "valid" => ConvolveMode::Valid,
        "same" => ConvolveMode::Same,
        _ => ConvolveMode::Full,
    };
    convolve(&a_arr, &b_arr, conv_mode).to_vec()
}

/// FFT-based convolution.
/// Modes: "full", "valid", "same".
pub fn fftconvolve(a: &[f64], b: &[f64], mode: &str) -> Vec<f64> {
    let n1 = a.len();
    let n2 = b.len();
    if n1 == 0 || n2 == 0 {
        return vec![];
    }
    let out_len = n1 + n2 - 1;
    let fft_len = next_power_of_2(out_len);

    // Zero-pad and FFT
    let mut a_pad = vec![rustfft::num_complex::Complex::new(0.0, 0.0); fft_len];
    for (i, &v) in a.iter().enumerate() {
        a_pad[i] = rustfft::num_complex::Complex::new(v, 0.0);
    }
    let mut b_pad = vec![rustfft::num_complex::Complex::new(0.0, 0.0); fft_len];
    for (i, &v) in b.iter().enumerate() {
        b_pad[i] = rustfft::num_complex::Complex::new(v, 0.0);
    }

    let mut planner = rustfft::FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    fft_fwd.process(&mut a_pad);
    fft_fwd.process(&mut b_pad);

    // Multiply in frequency domain
    for i in 0..fft_len {
        a_pad[i] = a_pad[i] * b_pad[i];
    }

    // Inverse FFT
    let fft_inv = planner.plan_fft_inverse(fft_len);
    fft_inv.process(&mut a_pad);
    let scale = 1.0 / fft_len as f64;
    let full: Vec<f64> = (0..out_len).map(|i| a_pad[i].re * scale).collect();

    match mode {
        "same" => {
            let start = (n2 - 1) / 2;
            full[start..start + n1].to_vec()
        }
        "valid" => {
            if n1 >= n2 {
                full[n2 - 1..n1].to_vec()
            } else {
                full[n1 - 1..n2].to_vec()
            }
        }
        _ => full,
    }
}

// ── Resampling ─────────────────────────────────────────────────────────────

/// Resample signal to `num` samples using FFT.
pub fn resample(x: &[f64], num: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || num == 0 {
        return vec![];
    }
    if num == n {
        return x.to_vec();
    }

    let spectrum = crate::sp::fft::rfft(x);
    let half = n / 2 + 1;

    if num > n {
        // Upsample: zero-pad spectrum
        let mut new_spectrum = vec![(0.0, 0.0); num / 2 + 1];
        let copy_len = half.min(num / 2 + 1);
        for i in 0..copy_len {
            new_spectrum[i] = spectrum[i];
        }
        crate::sp::fft::irfft(&new_spectrum, num)
    } else {
        // Downsample: truncate spectrum
        let new_half = num / 2 + 1;
        let mut new_spectrum = vec![(0.0, 0.0); new_half];
        let copy_len = new_half.min(half);
        for i in 0..copy_len {
            new_spectrum[i] = spectrum[i];
        }
        crate::sp::fft::irfft(&new_spectrum, num)
    }
}

/// Welch's method for power spectral density estimation.
/// Returns (frequencies, psd).
pub fn welch(x: &[f64], nperseg: usize, noverlap: usize) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let step = nperseg - noverlap;
    let window = hann_window(nperseg);
    let win_sum_sq: f64 = window.iter().map(|w| w * w).sum();

    let mut segments = Vec::new();
    let mut start = 0;
    while start + nperseg <= n {
        let mut seg = vec![0.0; nperseg];
        for i in 0..nperseg {
            seg[i] = x[start + i] * window[i];
        }
        segments.push(seg);
        start += step;
    }

    if segments.is_empty() {
        return (vec![], vec![]);
    }

    let nfft = nperseg;
    let psd_len = nfft / 2 + 1;
    let mut psd = vec![0.0; psd_len];

    for seg in &segments {
        let spectrum = crate::sp::fft::rfft(seg);
        for i in 0..psd_len {
            let re = spectrum[i].0;
            let im = spectrum[i].1;
            psd[i] += re * re + im * im;
        }
    }

    let num_segments = segments.len() as f64;
    let fs = 1.0;
    let freqs: Vec<f64> = (0..psd_len).map(|i| i as f64 * fs / nfft as f64).collect();

    for p in psd.iter_mut() {
        *p /= num_segments * win_sum_sq * fs;
    }

    (freqs, psd)
}

/// Periodogram of a single segment.
fn periodogram_segment(seg: &[f64]) -> Vec<f64> {
    let n = seg.len();
    let spectrum = crate::sp::fft::rfft(seg);
    let psd_len = n / 2 + 1;
    (0..psd_len)
        .map(|i| {
            let re = spectrum[i].0;
            let im = spectrum[i].1;
            (re * re + im * im) / n as f64
        })
        .collect()
}

/// Compute the spectrogram (time-frequency representation).
/// Returns (frequencies, time_slices of PSD).
pub fn spectrogram(x: &[f64], nperseg: usize, noverlap: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = x.len();
    let step = nperseg - noverlap;
    let window = hann_window(nperseg);

    let mut slices = Vec::new();
    let mut start = 0;
    while start + nperseg <= n {
        let mut seg = vec![0.0; nperseg];
        for i in 0..nperseg {
            seg[i] = x[start + i] * window[i];
        }
        slices.push(periodogram_segment(&seg));
        start += step;
    }

    let psd_len = nperseg / 2 + 1;
    let fs = 1.0;
    let freqs: Vec<f64> = (0..psd_len).map(|i| i as f64 * fs / nperseg as f64).collect();
    (freqs, slices)
}

/// Short-time Fourier transform.
/// Apply window, compute FFT of each segment.
pub fn stft(x: &[f64], nperseg: usize, noverlap: usize) -> Vec<Vec<(f64, f64)>> {
    let n = x.len();
    let step = nperseg - noverlap;
    let window = hann_window(nperseg);

    let mut result = Vec::new();
    let mut start = 0;
    while start + nperseg <= n {
        let mut seg = vec![0.0; nperseg];
        for i in 0..nperseg {
            seg[i] = x[start + i] * window[i];
        }
        result.push(crate::sp::fft::rfft(&seg));
        start += step;
    }
    result
}

fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if n <= 1 {
                1.0
            } else {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos())
            }
        })
        .collect()
}

// ── Filtering ──────────────────────────────────────────────────────────────

/// Savitzky-Golay smoothing filter.
/// Fit polynomial of order `polyorder` to each window, evaluate at center.
pub fn savgol_filter(x: &[f64], window_length: usize, polyorder: usize) -> Vec<f64> {
    let n = x.len();
    if window_length == 0 || window_length > n {
        return x.to_vec();
    }
    let coeffs = savgol_coeffs(window_length, polyorder);
    let half = window_length / 2;
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut val = 0.0;
        for (j, &c) in coeffs.iter().enumerate() {
            let idx = i as isize + j as isize - half as isize;
            let xval = if idx < 0 {
                x[0]
            } else if idx >= n as isize {
                x[n - 1]
            } else {
                x[idx as usize]
            };
            val += c * xval;
        }
        result[i] = val;
    }
    result
}

/// 1D median filter.
/// Replace each sample with median of neighborhood.
pub fn medfilt(x: &[f64], kernel_size: usize) -> Vec<f64> {
    let n = x.len();
    let half = kernel_size / 2;
    let mut result = vec![0.0; n];
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = (i + half + 1).min(n);
        let mut window: Vec<f64> = x[lo..hi].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result[i] = window[window.len() / 2];
    }
    result
}

/// Remove trend from signal.
/// "linear": subtract least-squares line. "constant": subtract mean.
pub fn detrend(x: &[f64], type_: &str) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    if type_ == "constant" {
        let mean: f64 = x.iter().sum::<f64>() / n as f64;
        return x.iter().map(|&v| v - mean).collect();
    }
    // Linear detrend: fit y = a + b*x via least squares
    let xvals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let sx: f64 = xvals.iter().sum();
    let sy: f64 = x.iter().sum();
    let sxx: f64 = xvals.iter().map(|v| v * v).sum();
    let sxy: f64 = xvals.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    let denom = n as f64 * sxx - sx * sx;
    let b = (n as f64 * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / n as f64;
    xvals.iter().zip(x.iter()).map(|(&xi, &yi)| yi - (a + b * xi)).collect()
}

// ── Analytic signal ───────────────────────────────────────────────────────

/// Analytic signal via Hilbert transform.
/// FFT, zero negative frequencies, double positive frequencies, IFFT.
/// Returns complex (real, imag) pairs.
pub fn hilbert(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    let spectrum = crate::sp::fft::fft_real(&Array1::from_vec(x.to_vec()));
    let mut h = vec![rustfft::num_complex::Complex::new(0.0, 0.0); n];
    if n % 2 == 0 {
        h[0] = rustfft::num_complex::Complex::new(1.0, 0.0);
        for i in 1..n / 2 {
            h[i] = rustfft::num_complex::Complex::new(2.0, 0.0);
        }
        h[n / 2] = rustfft::num_complex::Complex::new(1.0, 0.0);
    } else {
        h[0] = rustfft::num_complex::Complex::new(1.0, 0.0);
        for i in 1..(n + 1) / 2 {
            h[i] = rustfft::num_complex::Complex::new(2.0, 0.0);
        }
    }
    let mut analytic_spectrum = spectrum;
    for i in 0..n {
        analytic_spectrum[i] = analytic_spectrum[i] * h[i];
    }
    let result = crate::sp::fft::ifft(&analytic_spectrum);
    result.iter().map(|c| (c.re, c.im)).collect()
}

// ── Peak analysis ──────────────────────────────────────────────────────────

/// Prominence of each peak.
/// Prominence = peak_height - max(left_base, right_base)
/// where bases are the lowest points between peak and next higher peak.
pub fn peak_prominences(x: &[f64], peaks: &[usize]) -> Vec<f64> {
    let n = x.len();
    peaks
        .iter()
        .map(|&pk| {
            let pk_height = x[pk];
            // Find left base: lowest point going left until a higher peak or start
            let mut left_base = x[pk];
            for j in (0..pk).rev() {
                if x[j] < left_base {
                    left_base = x[j];
                }
                if x[j] > pk_height {
                    break;
                }
            }
            // Find right base: lowest point going right until a higher peak or end
            let mut right_base = x[pk];
            for j in pk + 1..n {
                if x[j] < right_base {
                    right_base = x[j];
                }
                if x[j] > pk_height {
                    break;
                }
            }
            pk_height - left_base.max(right_base)
        })
        .collect()
}

/// Widths of peaks at `rel_height` * prominence.
/// For each peak, find left/right crossing points.
pub fn peak_widths(x: &[f64], peaks: &[usize], rel_height: f64) -> Vec<f64> {
    let prominences = peak_prominences(x, peaks);
    let n = x.len();

    peaks
        .iter()
        .zip(prominences.iter())
        .map(|(&pk, &prom)| {
            let threshold = x[pk] - rel_height * prom;
            // Find left crossing
            let mut left = pk;
            for j in (0..pk).rev() {
                if x[j] <= threshold {
                    left = j;
                    break;
                }
                if j == 0 {
                    left = 0;
                }
            }
            // Find right crossing
            let mut right = pk;
            for j in pk + 1..n {
                if x[j] <= threshold {
                    right = j;
                    break;
                }
                if j == n - 1 {
                    right = n - 1;
                }
            }
            (right - left) as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butter_lowpass_basic() {
        let (b, a, gain) = butter(2, 0.5, "low");
        assert_eq!(b.len(), 3);
        assert_eq!(a.len(), 3);
        assert!(gain > 0.0);
        // a[0] should be 1 (monic denominator after scaling)
        assert!((a[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_butter_highpass_basic() {
        let (b, a, gain) = butter(2, 0.5, "high");
        assert_eq!(b.len(), 3);
        assert_eq!(a.len(), 3);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_butter_lowpass_dc_gain() {
        // Lowpass Butterworth should have unity gain at DC
        let (b, a, _) = butter(4, 0.3, "low");
        let (_, mags) = freqz(&b, &a, 512);
        // DC gain (first bin) should be ~1.0
        assert!((mags[0] - 1.0).abs() < 0.01, "DC gain = {}", mags[0]);
    }

    #[test]
    fn test_cheby1_basic() {
        let (b, a, gain) = cheby1(3, 1.0, 0.5, "low");
        assert_eq!(b.len(), 4);
        assert_eq!(a.len(), 4);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_cheby1_ripple() {
        // Chebyshev Type I with even order has DC gain = 1/sqrt(1+eps^2)
        // For rp=1dB, that's ~0.891
        let (b, a, _) = cheby1(4, 1.0, 0.3, "low");
        let (_, mags) = freqz(&b, &a, 1024);
        assert!(mags[0] > 0.8 && mags[0] <= 1.0, "DC gain = {}", mags[0]);
    }

    #[test]
    fn test_sosfilt_identity() {
        // Identity filter: b=[1], a=[1]
        let sos = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = sosfilt(&sos, &x);
        for i in 0..x.len() {
            assert!((y[i] - x[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sosfilt_gain() {
        // Gain of 2.0
        let sos = [[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let x = vec![1.0, 2.0, 3.0];
        let y = sosfilt(&sos, &x);
        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[1] - 4.0).abs() < 1e-10);
        assert!((y[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sosfiltfilt_symmetry() {
        // Use a symmetric filter (averaging) so zero-phase output is symmetric
        let sos = [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 0.0, 0.0]];
        let x = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let y = sosfiltfilt(&sos, &x);
        // Symmetric filter + symmetric input -> symmetric output
        assert!((y[0] - y[4]).abs() < 1e-10);
        assert!((y[1] - y[3]).abs() < 1e-10);
    }

    #[test]
    fn test_sosfiltfilt_impulse() {
        let (b, a, _) = butter(2, 0.3, "low");
        let sos = [[b[0], b[1], b[2], a[0], a[1], a[2]]];
        let mut x = vec![0.0; 100];
        x[50] = 1.0;
        let y = sosfiltfilt(&sos, &x);
        // Peak should be at center
        let peak = y.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!(peak >= 48 && peak <= 52, "peak at {}", peak);
    }

    #[test]
    fn test_freqz_length() {
        let b = vec![1.0, 0.5];
        let a = vec![1.0, -0.2];
        let (freqs, mags) = freqz(&b, &a, 256);
        assert_eq!(freqs.len(), 256);
        assert_eq!(mags.len(), 256);
    }

    #[test]
    fn test_freqz_dc_gain() {
        // Allpass filter: b=[1], a=[1] -> DC gain = 1
        let b = vec![1.0];
        let a = vec![1.0];
        let (_, mags) = freqz(&b, &a, 128);
        assert!((mags[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlate_full() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = correlate(&a, &b, "full");
        assert_eq!(c.len(), 5);
        // Autocorrelation peak should be at center
        let max_idx = c.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_correlate_same() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0];
        let c = correlate(&a, &b, "same");
        assert_eq!(c.len(), 5);
    }

    #[test]
    fn test_fftconvolve_full() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let c = fftconvolve(&a, &b, "full");
        assert_eq!(c.len(), 5);
        assert!((c[0] - 0.0).abs() < 1e-10);
        assert!((c[1] - 1.0).abs() < 1e-10);
        assert!((c[2] - 2.5).abs() < 1e-10);
        assert!((c[3] - 4.0).abs() < 1e-10);
        assert!((c[4] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_fftconvolve_same() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0];
        let c = fftconvolve(&a, &b, "same");
        assert_eq!(c.len(), 4);
    }

    #[test]
    fn test_resample_upsample() {
        let x = vec![0.0, 1.0, 0.0, -1.0];
        let y = resample(&x, 8);
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_resample_downsample() {
        let x: Vec<f64> = (0..16).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin()).collect();
        let y = resample(&x, 8);
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_resample_roundtrip() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = resample(&x, 4);
        assert_eq!(y.len(), 4);
        for i in 0..4 {
            assert!((y[i] - x[i]).abs() < 1e-10, "y[{}] = {}, x[{}] = {}", i, y[i], i, x[i]);
        }
    }

    #[test]
    fn test_welch_basic() {
        let x: Vec<f64> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * 4.0 * i as f64 / 256.0).sin())
            .collect();
        let (freqs, psd) = welch(&x, 64, 32);
        assert!(!freqs.is_empty());
        assert!(!psd.is_empty());
        assert_eq!(freqs.len(), psd.len());
    }

    #[test]
    fn test_welch_peak_frequency() {
        let n = 512;
        let freq = 8.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin())
            .collect();
        let (freqs, psd) = welch(&x, 128, 64);
        // Find peak in PSD (skip DC)
        let peak_idx = psd[1..]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
            + 1;
        // Peak frequency should be near 8/512 = 0.015625
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - freq / n as f64).abs() < 0.01,
            "peak_freq = {}",
            peak_freq
        );
    }

    #[test]
    fn test_spectrogram_basic() {
        let x: Vec<f64> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * 4.0 * i as f64 / 256.0).sin())
            .collect();
        let (freqs, slices) = spectrogram(&x, 64, 32);
        assert!(!freqs.is_empty());
        assert!(!slices.is_empty());
        assert_eq!(freqs.len(), slices[0].len());
    }

    #[test]
    fn test_stft_basic() {
        let x: Vec<f64> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * 4.0 * i as f64 / 256.0).sin())
            .collect();
        let result = stft(&x, 64, 32);
        assert!(!result.is_empty());
        assert_eq!(result[0].len(), 33); // 64/2 + 1
    }

    #[test]
    fn test_savgol_filter_smoothing() {
        // Noisy sine wave
        let x: Vec<f64> = (0..100)
            .map(|i| {
                (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin()
                    + 0.1 * (i as f64 * 0.7).sin()
            })
            .collect();
        let y = savgol_filter(&x, 11, 3);
        assert_eq!(y.len(), x.len());
        // Filtered signal should be smoother (lower total variation)
        let var_orig: f64 = (1..x.len()).map(|i| (x[i] - x[i - 1]).powi(2)).sum();
        let var_filt: f64 = (1..y.len()).map(|i| (y[i] - y[i - 1]).powi(2)).sum();
        assert!(var_filt <= var_orig);
    }

    #[test]
    fn test_savgol_filter_identity() {
        // Polynomial of order <= polyorder should pass through unchanged
        // y = 1 + 2x + 3x^2
        let x: Vec<f64> = (0..21).map(|i| 1.0 + 2.0 * i as f64 + 3.0 * (i as f64).powi(2)).collect();
        let y = savgol_filter(&x, 7, 2);
        for i in 3..18 {
            assert!(
                (y[i] - x[i]).abs() < 1e-6,
                "y[{}] = {}, x[{}] = {}",
                i,
                y[i],
                i,
                x[i]
            );
        }
    }

    #[test]
    fn test_medfilt_basic() {
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let y = medfilt(&x, 3);
        assert_eq!(y.len(), 5);
        // i=0: window=[1,3] -> median=3; i=1: [1,3,2]->2; i=2: [3,2,5]->3; i=3: [2,5,4]->4; i=4: [5,4]->5
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
        assert!((y[2] - 3.0).abs() < 1e-10);
        assert!((y[3] - 4.0).abs() < 1e-10);
        assert!((y[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_medfilt_removes_spikes() {
        let mut x = vec![1.0; 100];
        x[50] = 100.0; // spike
        let y = medfilt(&x, 5);
        // Spike should be removed
        assert!((y[50] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detrend_constant() {
        let x = vec![5.0, 7.0, 9.0, 11.0, 13.0];
        let y = detrend(&x, "constant");
        let mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_detrend_linear() {
        let x: Vec<f64> = (0..100).map(|i| 2.0 + 3.0 * i as f64).collect();
        let y = detrend(&x, "linear");
        // Should be near zero
        for v in &y {
            assert!(v.abs() < 1e-10, "detrend residual = {}", v);
        }
    }

    #[test]
    fn test_hilbert_cosine() {
        // Analytic signal of cos(wt) should have |magnitude| ~ 1
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 4.0 * i as f64 / n as f64).cos())
            .collect();
        let analytic = hilbert(&x);
        // Magnitude should be close to 1 for most points (away from edges)
        for i in 20..n - 20 {
            let (re, im) = analytic[i];
            let mag = (re * re + im * im).sqrt();
            assert!(
                (mag - 1.0).abs() < 0.15,
                "magnitude at {} = {}",
                i,
                mag
            );
        }
    }

    #[test]
    fn test_hilbert_sine_phase() {
        // Analytic signal of sin(wt) should have ~constant phase
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let analytic = hilbert(&x);
        // For sin, analytic signal ~ -j*e^{jwt}, so imag should be ~ -cos
        // Check that real part is close to sin
        for i in 20..n - 20 {
            let (re, _im) = analytic[i];
            let expected = x[i];
            assert!(
                (re - expected).abs() < 0.15,
                "real part at {}: {} vs {}",
                i,
                re,
                expected
            );
        }
    }

    #[test]
    fn test_peak_prominences_basic() {
        let x = vec![0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0];
        let peaks = vec![1, 3, 5];
        let proms = peak_prominences(&x, &peaks);
        assert_eq!(proms.len(), 3);
        // Peak at index 3 (height 3): bases are 0 on both sides -> prominence = 3
        assert!((proms[1] - 3.0).abs() < 1e-10, "prominence[1] = {}", proms[1]);
    }

    #[test]
    fn test_peak_prominences_asymmetric() {
        let x = vec![0.0, 2.0, 1.0, 5.0, 2.0, 0.0];
        let peaks = vec![1, 3];
        let proms = peak_prominences(&x, &peaks);
        // Peak at 1 (height 2): right base = 1, prominence = 2-1 = 1
        assert!((proms[0] - 1.0).abs() < 1e-10, "prom[0] = {}", proms[0]);
        // Peak at 3 (height 5): bases = 0 on both sides, prominence = 5
        assert!((proms[1] - 5.0).abs() < 1e-10, "prom[1] = {}", proms[1]);
    }

    #[test]
    fn test_peak_widths_basic() {
        // Triangle peak
        let x = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let peaks = vec![3];
        let widths = peak_widths(&x, &peaks, 0.5);
        assert_eq!(widths.len(), 1);
        assert!(widths[0] > 0.0);
    }

    #[test]
    fn test_peak_widths_narrow_peak() {
        let mut x = vec![0.0; 21];
        x[10] = 5.0;
        x[9] = 1.0;
        x[11] = 1.0;
        let peaks = vec![10];
        let widths = peak_widths(&x, &peaks, 0.5);
        assert_eq!(widths.len(), 1);
        // Width should be small
        assert!(widths[0] < 10.0, "width = {}", widths[0]);
    }
}

