use ndarray::Array2;
use rand::seq::SliceRandom;

/// Compute minibatch indices for a dataset.
///
/// Returns (indices_batches, n_batches).
pub fn minibatch_indices(n_samples: usize, batch_size: usize, shuffle: bool) -> (Vec<Vec<usize>>, usize) {
    let n_batches = (n_samples + batch_size - 1) / batch_size;
    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
    }

    let mut batches = Vec::with_capacity(n_batches);
    for i in 0..n_batches {
        let start = i * batch_size;
        let end = (start + batch_size).min(n_samples);
        batches.push(indices[start..end].to_vec());
    }

    (batches, n_batches)
}

/// 1D convolution using im2col-style approach.
pub fn conv1d_forward(
    x: &Array2<f64>,
    w: &Array2<f64>,
    stride: usize,
    pad: usize,
) -> Array2<f64> {
    let (n_ex, in_len) = x.dim();
    let kernel_width = w.nrows();

    let padded_len = in_len + 2 * pad;
    let out_len = (padded_len - kernel_width) / stride + 1;

    // Pad input
    let mut x_padded = Array2::zeros((n_ex, padded_len));
    for m in 0..n_ex {
        for i in 0..in_len {
            x_padded[[m, i + pad]] = x[[m, i]];
        }
    }

    // Compute convolution
    let mut output = Array2::zeros((n_ex, out_len));
    for m in 0..n_ex {
        for i in 0..out_len {
            let mut sum = 0.0;
            for k in 0..kernel_width {
                sum += x_padded[[m, i * stride + k]] * w[[k, 0]];
            }
            output[[m, i]] = sum;
        }
    }
    output
}

/// 2D convolution: cross-correlation of input X with weight volume W.
///
/// X shape: (n_ex, in_rows, in_cols, in_ch)
/// W shape: (fr, fc, in_ch, out_ch)
/// Output shape: (n_ex, out_rows, out_cols, out_ch)
pub fn conv2d_forward(
    x: &Array2<f64>,
    w: &Array2<f64>,
    stride: usize,
    pad: usize,
) -> Array2<f64> {
    let (n_ex, in_rows) = x.dim();
    let in_cols = in_rows; // Assume square for simplicity in 2D case
    let (fr, fc) = (w.nrows(), w.ncols());

    let out_rows = (in_rows + 2 * pad - fr) / stride + 1;
    let out_cols = (in_cols + 2 * pad - fc) / stride + 1;

    // Pad input
    let padded_rows = in_rows + 2 * pad;
    let padded_cols = in_cols + 2 * pad;
    let mut x_padded = Array2::zeros((n_ex, padded_rows * padded_cols));
    for m in 0..n_ex {
        for r in 0..in_rows {
            for c in 0..in_cols {
                x_padded[[m, (r + pad) * padded_cols + (c + pad)]] = x[[m, r * in_cols + c]];
            }
        }
    }

    // Compute convolution
    let mut output = Array2::zeros((n_ex, out_rows * out_cols));
    for m in 0..n_ex {
        for i in 0..out_rows {
            for j in 0..out_cols {
                let mut sum = 0.0;
                for fi in 0..fr {
                    for fj in 0..fc {
                        let pr = i * stride + fi;
                        let pc = j * stride + fj;
                        sum += x_padded[[m, pr * padded_cols + pc]] * w[[fi * fc + fj, 0]];
                    }
                }
                output[[m, i * out_cols + j]] = sum;
            }
        }
    }
    output
}

/// Max pooling for 1D input.
pub fn max_pool1d(x: &Array2<f64>, kernel_width: usize, stride: usize) -> Array2<f64> {
    let (n_ex, in_len) = x.dim();
    let out_len = (in_len - kernel_width) / stride + 1;

    let mut output = Array2::zeros((n_ex, out_len));
    for m in 0..n_ex {
        for i in 0..out_len {
            let mut max_val = f64::NEG_INFINITY;
            for k in 0..kernel_width {
                let idx = i * stride + k;
                if idx < in_len {
                    max_val = max_val.max(x[[m, idx]]);
                }
            }
            output[[m, i]] = max_val;
        }
    }
    output
}

/// Average pooling for 1D input.
pub fn avg_pool1d(x: &Array2<f64>, kernel_width: usize, stride: usize) -> Array2<f64> {
    let (n_ex, in_len) = x.dim();
    let out_len = (in_len - kernel_width) / stride + 1;

    let mut output = Array2::zeros((n_ex, out_len));
    for m in 0..n_ex {
        for i in 0..out_len {
            let mut sum = 0.0;
            let mut count = 0;
            for k in 0..kernel_width {
                let idx = i * stride + k;
                if idx < in_len {
                    sum += x[[m, idx]];
                    count += 1;
                }
            }
            output[[m, i]] = sum / count as f64;
        }
    }
    output
}

/// Softmax activation along axis 1.
pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut output = x.clone();
    let n_rows = x.nrows();
    let n_cols = x.ncols();

    for i in 0..n_rows {
        let max_val = (0..n_cols).map(|j| x[[i, j]]).fold(f64::NEG_INFINITY, f64::max);
        let mut exp_sum = 0.0;
        for j in 0..n_cols {
            output[[i, j]] = (x[[i, j]] - max_val).exp();
            exp_sum += output[[i, j]];
        }
        for j in 0..n_cols {
            output[[i, j]] /= exp_sum;
        }
    }
    output
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_minibatch_indices() {
        let (batches, n) = minibatch_indices(100, 32, false);
        assert_eq!(n, 4);
        assert_eq!(batches[0].len(), 32);
        assert_eq!(batches[3].len(), 4);
    }

    #[test]
    fn test_softmax() {
        let x = array![[1.0, 2.0, 3.0]];
        let y = softmax(&x);
        let sum: f64 = y.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(y[[0, 2]] > y[[0, 1]]);
        assert!(y[[0, 1]] > y[[0, 0]]);
    }

    #[test]
    fn test_max_pool1d() {
        let x = array![[1.0, 3.0, 2.0, 4.0, 1.0, 2.0]];
        let y = max_pool1d(&x, 2, 2);
        assert_eq!(y, array![[3.0, 4.0, 2.0]]);
    }
}
