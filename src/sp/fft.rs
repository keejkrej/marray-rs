use ndarray::{Array1, Array2};
pub use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// 1D Forward Fast Fourier Transform.
pub fn fft(arr: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let n = arr.len();
    let fft_op = planner.plan_fft_forward(n);
    let mut buffer = arr.to_vec();
    fft_op.process(&mut buffer);
    Array1::from_vec(buffer)
}

/// 1D Inverse Fast Fourier Transform.
pub fn ifft(arr: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let n = arr.len();
    let fft_op = planner.plan_fft_inverse(n);
    let mut buffer = arr.to_vec();
    fft_op.process(&mut buffer);
    let scale = 1.0 / (n as f64);
    for val in &mut buffer {
        *val *= scale;
    }
    Array1::from_vec(buffer)
}

/// Helper for running FFT on real vectors.
pub fn fft_real(arr: &Array1<f64>) -> Array1<Complex<f64>> {
    let complex_arr = arr.mapv(|x| Complex::new(x, 0.0));
    fft(&complex_arr)
}

/// 2D Forward Fast Fourier Transform.
pub fn fft2(arr: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (nrows, ncols) = arr.dim();
    let mut temp = Array2::zeros((nrows, ncols));
    let mut planner = FftPlanner::new();

    let row_fft = planner.plan_fft_forward(ncols);
    for r in 0..nrows {
        let mut row_buf = arr.row(r).to_vec();
        row_fft.process(&mut row_buf);
        for c in 0..ncols {
            temp[[r, c]] = row_buf[c];
        }
    }

    let col_fft = planner.plan_fft_forward(nrows);
    let mut result = Array2::zeros((nrows, ncols));
    for c in 0..ncols {
        let mut col_buf = temp.column(c).to_vec();
        col_fft.process(&mut col_buf);
        for r in 0..nrows {
            result[[r, c]] = col_buf[r];
        }
    }

    result
}

/// 2D Inverse Fast Fourier Transform.
pub fn ifft2(arr: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (nrows, ncols) = arr.dim();
    let mut temp = Array2::zeros((nrows, ncols));
    let mut planner = FftPlanner::new();

    let row_ifft = planner.plan_fft_inverse(ncols);
    for r in 0..nrows {
        let mut row_buf = arr.row(r).to_vec();
        row_ifft.process(&mut row_buf);
        for c in 0..ncols {
            temp[[r, c]] = row_buf[c];
        }
    }

    let col_ifft = planner.plan_fft_inverse(nrows);
    let mut result = Array2::zeros((nrows, ncols));
    for c in 0..ncols {
        let mut col_buf = temp.column(c).to_vec();
        col_ifft.process(&mut col_buf);
        for r in 0..nrows {
            result[[r, c]] = col_buf[r];
        }
    }

    let scale = 1.0 / ((nrows * ncols) as f64);
    result.mapv(|x| x * scale)
}

/// 1D Real FFT returning only non-negative frequency components.
/// Output length is n/2 + 1 where n = x.len().
pub fn rfft(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    let arr = Array1::from_vec(x.to_vec());
    let spectrum = fft_real(&arr);
    let out_len = n / 2 + 1;
    (0..out_len)
        .map(|i| (spectrum[i].re, spectrum[i].im))
        .collect()
}

/// Inverse of rfft. Reconstructs full complex spectrum from half-spectrum, then IFFT.
pub fn irfft(spectrum: &[(f64, f64)], n: usize) -> Vec<f64> {
    let half = spectrum.len();
    let mut full = Vec::with_capacity(n);
    for &(re, im) in spectrum {
        full.push(Complex::new(re, im));
    }
    for i in half..n {
        let mirror = n - i;
        if mirror < half {
            full.push(Complex::new(spectrum[mirror].0, -spectrum[mirror].1));
        } else {
            full.push(Complex::new(0.0, 0.0));
        }
    }
    let arr = Array1::from_vec(full);
    let result = ifft(&arr);
    result.iter().map(|c| c.re).collect()
}

/// Sample frequencies for FFT.
/// Returns [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n).
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let mut freqs = Vec::with_capacity(n);
    let scale = 1.0 / (d * n as f64);
    let mid = n.div_ceil(2);
    for i in 0..mid {
        freqs.push(i as f64 * scale);
    }
    for i in mid..n {
        freqs.push((i as f64 - n as f64) * scale);
    }
    freqs
}

/// Sample frequencies for rfft.
/// Returns [0, 1, ..., n/2] / (d*n).
pub fn rfftfreq(n: usize, d: f64) -> Vec<f64> {
    let out_len = n / 2 + 1;
    let scale = 1.0 / (d * n as f64);
    (0..out_len).map(|i| i as f64 * scale).collect()
}

/// Shift zero-frequency component to center.
/// For even n, swap first and second halves.
/// For odd n, swap with the middle element going to the end.
pub fn fftshift<T: Clone>(x: &[T]) -> Vec<T> {
    let n = x.len();
    let mid = n.div_ceil(2);
    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&x[mid..]);
    result.extend_from_slice(&x[..mid]);
    result
}

/// Inverse of fftshift.
pub fn ifftshift<T: Clone>(x: &[T]) -> Vec<T> {
    let n = x.len();
    let mid = n / 2;
    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&x[mid..]);
    result.extend_from_slice(&x[..mid]);
    result
}

/// 2D Real FFT. Apply rfft along rows, then along columns.
pub fn rfft2(x: &Array2<f64>) -> Array2<(f64, f64)> {
    let (nrows, ncols) = x.dim();
    let out_cols = ncols / 2 + 1;

    let mut temp = vec![vec![(0.0, 0.0); out_cols]; nrows];
    for r in 0..nrows {
        let row_vec: Vec<f64> = x.row(r).to_vec();
        let row_fft = rfft(&row_vec);
        for c in 0..out_cols {
            temp[r][c] = row_fft[c];
        }
    }

    let mut result = vec![vec![(0.0, 0.0); out_cols]; nrows];
    for c in 0..out_cols {
        let col_re: Vec<f64> = (0..nrows).map(|r| temp[r][c].0).collect();
        let col_im: Vec<f64> = (0..nrows).map(|r| temp[r][c].1).collect();
        let fft_re = rfft(&col_re);
        let fft_im = rfft(&col_im);
        for r in 0..nrows {
            result[r][c] = (fft_re[r].0 - fft_im[r].1, fft_re[r].1 + fft_im[r].0);
        }
    }

    
    Array2::from_shape_fn((nrows, out_cols), |(r, c)| result[r][c])
}

/// 2D Inverse Real FFT.
pub fn irfft2(spectrum: &Array2<(f64, f64)>, shape: (usize, usize)) -> Array2<f64> {
    let (_, out_cols) = spectrum.dim();
    let (nrows, ncols) = shape;

    let mut temp = vec![vec![0.0; out_cols]; nrows];
    for c in 0..out_cols {
        let col_vec: Vec<(f64, f64)> = (0..nrows).map(|r| spectrum[[r, c]]).collect();
        let col_result = irfft(&col_vec, nrows);
        for r in 0..nrows {
            temp[r][c] = col_result[r];
        }
    }

    let mut result = Array2::zeros((nrows, ncols));
    for r in 0..nrows {
        let row_vec: Vec<(f64, f64)> = (0..out_cols).map(|c| (temp[r][c], 0.0)).collect();
        let row_result = irfft(&row_vec, ncols);
        for c in 0..ncols {
            result[[r, c]] = row_result[c];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fftfreq() {
        let freqs = fftfreq(8, 1.0);
        let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];
        for (a, b) in freqs.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "fftfreq: {} != {}", a, b);
        }
    }

    #[test]
    fn test_rfftfreq() {
        let freqs = rfftfreq(8, 1.0);
        let expected = [0.0, 0.125, 0.25, 0.375, 0.5];
        assert_eq!(freqs.len(), expected.len());
        for (a, b) in freqs.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "rfftfreq: {} != {}", a, b);
        }
    }

    #[test]
    fn test_fftshift_even() {
        let result = fftshift(&[0, 1, 2, 3]);
        assert_eq!(result, vec![2, 3, 0, 1]);
    }

    #[test]
    fn test_fftshift_odd() {
        let result = fftshift(&[0, 1, 2, 3, 4]);
        assert_eq!(result, vec![3, 4, 0, 1, 2]);
    }

    #[test]
    fn test_fftshift_roundtrip() {
        let x = vec![10, 20, 30, 40, 50];
        let shifted = fftshift(&x);
        let unshifted = ifftshift(&shifted);
        assert_eq!(x, unshifted);

        let x_even = vec![1, 2, 3, 4];
        let shifted = fftshift(&x_even);
        let unshifted = ifftshift(&shifted);
        assert_eq!(x_even, unshifted);
    }

    #[test]
    fn test_rfft_cosine() {
        let n = 64;
        let freq = 4.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).cos())
            .collect();
        let spectrum = rfft(&x);
        let mags: Vec<f64> = spectrum.iter().map(|(re, im)| (re * re + im * im).sqrt()).collect();
        let peak = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 4, "rfft peak should be at frequency index 4, got {}", peak);
    }

    #[test]
    fn test_rfft_irfft_roundtrip() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = rfft(&x);
        let reconstructed = irfft(&spectrum, x.len());
        for (a, b) in x.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10, "irfft roundtrip: {} != {}", a, b);
        }
    }
}
