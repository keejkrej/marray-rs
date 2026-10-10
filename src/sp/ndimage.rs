use ndarray::{Array1, Array2};

/// 1D uniform (box) filter / moving average.
pub fn rsuniform_filter1d(input: &Array1<f64>, size: usize) -> Array1<f64> {
    assert!(size > 0, "size must be positive");
    let n = input.len();
    let mut out = Array1::zeros(n);
    for i in 0..n {
        let start = i;
        let end = (i + size).min(n);
        let window = &input.slice(ndarray::s![start..end]);
        out[i] = window.sum() / (end - start) as f64;
    }
    out
}

/// 1D Gaussian filter with reflection boundary handling.
pub fn gaussian_filter1d(input: &[f64], sigma: f64) -> Vec<f64> {
    assert!(sigma > 0.0, "sigma must be positive");
    let radius = (3.0 * sigma).ceil() as usize;
    let n = input.len();
    let mut kernel = Vec::with_capacity(2 * radius + 1);
    let mut sum = 0.0;
    for i in 0..=2 * radius {
        let x = i as f64 - radius as f64;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }
    let mut output = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..kernel.len() {
            let idx = i as isize + j as isize - radius as isize;
            let idx = if idx < 0 {
                (-idx) as usize
            } else if idx >= n as isize {
                2 * n - 1 - idx as usize
            } else {
                idx as usize
            };
            acc += input[idx] * kernel[j];
        }
        output[i] = acc;
    }
    output
}

/// 2D Gaussian blur. Separable: apply 1D Gaussian along rows, then columns.
pub fn gaussian_filter(input: &Array2<f64>, sigma: f64) -> Array2<f64> {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut temp = Array2::zeros((rows, cols));
    let mut output = Array2::zeros((rows, cols));
    for i in 0..rows {
        let row: Vec<f64> = input.row(i).to_vec();
        let filtered = gaussian_filter1d(&row, sigma);
        for j in 0..cols {
            temp[[i, j]] = filtered[j];
        }
    }
    for j in 0..cols {
        let col: Vec<f64> = temp.column(j).to_vec();
        let filtered = gaussian_filter1d(&col, sigma);
        for i in 0..rows {
            output[[i, j]] = filtered[i];
        }
    }
    output
}

/// 2D box filter (moving average).
pub fn uniform_filter(input: &Array2<f64>, size: usize) -> Array2<f64> {
    assert!(size > 0, "size must be positive");
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut output = Array2::zeros((rows, cols));
    let half = size / 2;
    for i in 0..rows {
        for j in 0..cols {
            let r_start = i.saturating_sub(half);
            let r_end = (i + half + 1).min(rows);
            let c_start = j.saturating_sub(half);
            let c_end = (j + half + 1).min(cols);
            let mut sum = 0.0;
            let mut count = 0;
            for r in r_start..r_end {
                for c in c_start..c_end {
                    sum += input[[r, c]];
                    count += 1;
                }
            }
            output[[i, j]] = sum / count as f64;
        }
    }
    output
}

/// 2D median filter.
pub fn median_filter(input: &Array2<f64>, size: usize) -> Array2<f64> {
    assert!(size > 0, "size must be positive");
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut output = Array2::zeros((rows, cols));
    let half = size / 2;
    for i in 0..rows {
        for j in 0..cols {
            let r_start = i.saturating_sub(half);
            let r_end = (i + half + 1).min(rows);
            let c_start = j.saturating_sub(half);
            let c_end = (j + half + 1).min(cols);
            let mut vals = Vec::new();
            for r in r_start..r_end {
                for c in c_start..c_end {
                    vals.push(input[[r, c]]);
                }
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            output[[i, j]] = vals[vals.len() / 2];
        }
    }
    output
}

/// 2D maximum filter.
pub fn maximum_filter(input: &Array2<f64>, size: usize) -> Array2<f64> {
    assert!(size > 0, "size must be positive");
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut output = Array2::zeros((rows, cols));
    let half = size / 2;
    for i in 0..rows {
        for j in 0..cols {
            let r_start = i.saturating_sub(half);
            let r_end = (i + half + 1).min(rows);
            let c_start = j.saturating_sub(half);
            let c_end = (j + half + 1).min(cols);
            let mut max_val = f64::NEG_INFINITY;
            for r in r_start..r_end {
                for c in c_start..c_end {
                    if input[[r, c]] > max_val {
                        max_val = input[[r, c]];
                    }
                }
            }
            output[[i, j]] = max_val;
        }
    }
    output
}

/// 2D minimum filter.
pub fn minimum_filter(input: &Array2<f64>, size: usize) -> Array2<f64> {
    assert!(size > 0, "size must be positive");
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut output = Array2::zeros((rows, cols));
    let half = size / 2;
    for i in 0..rows {
        for j in 0..cols {
            let r_start = i.saturating_sub(half);
            let r_end = (i + half + 1).min(rows);
            let c_start = j.saturating_sub(half);
            let c_end = (j + half + 1).min(cols);
            let mut min_val = f64::INFINITY;
            for r in r_start..r_end {
                for c in c_start..c_end {
                    if input[[r, c]] < min_val {
                        min_val = input[[r, c]];
                    }
                }
            }
            output[[i, j]] = min_val;
        }
    }
    output
}

/// Erode binary image. Output pixel is true only if ALL structure pixels under it are true.
pub fn binary_erosion(input: &Array2<bool>, structure: &Array2<bool>) -> Array2<bool> {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s_shape = structure.shape();
    let (s_rows, s_cols) = (s_shape[0], s_shape[1]);
    let s_center_r = s_rows / 2;
    let s_center_c = s_cols / 2;
    let mut output = Array2::from_elem((rows, cols), false);
    for i in 0..rows {
        for j in 0..cols {
            let mut all_true = true;
            for si in 0..s_rows {
                for sj in 0..s_cols {
                    if !structure[[si, sj]] {
                        continue;
                    }
                    let ni = i as isize + si as isize - s_center_r as isize;
                    let nj = j as isize + sj as isize - s_center_c as isize;
                    if ni < 0 || ni >= rows as isize || nj < 0 || nj >= cols as isize {
                        all_true = false;
                        break;
                    }
                    if !input[[ni as usize, nj as usize]] {
                        all_true = false;
                        break;
                    }
                }
                if !all_true {
                    break;
                }
            }
            output[[i, j]] = all_true;
        }
    }
    output
}

/// Dilate binary image. Output pixel is true if ANY structure pixel overlaps a true input pixel.
pub fn binary_dilation(input: &Array2<bool>, structure: &Array2<bool>) -> Array2<bool> {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s_shape = structure.shape();
    let (s_rows, s_cols) = (s_shape[0], s_shape[1]);
    let s_center_r = s_rows / 2;
    let s_center_c = s_cols / 2;
    let mut output = Array2::from_elem((rows, cols), false);
    for i in 0..rows {
        for j in 0..cols {
            let mut any_true = false;
            for si in 0..s_rows {
                for sj in 0..s_cols {
                    if !structure[[si, sj]] {
                        continue;
                    }
                    let ni = i as isize + si as isize - s_center_r as isize;
                    let nj = j as isize + sj as isize - s_center_c as isize;
                    if ni < 0 || ni >= rows as isize || nj < 0 || nj >= cols as isize {
                        continue;
                    }
                    if input[[ni as usize, nj as usize]] {
                        any_true = true;
                        break;
                    }
                }
                if any_true {
                    break;
                }
            }
            output[[i, j]] = any_true;
        }
    }
    output
}

/// Binary opening: erosion then dilation.
pub fn binary_opening(input: &Array2<bool>, structure: &Array2<bool>) -> Array2<bool> {
    binary_dilation(&binary_erosion(input, structure), structure)
}

/// Binary closing: dilation then erosion.
pub fn binary_closing(input: &Array2<bool>, structure: &Array2<bool>) -> Array2<bool> {
    binary_erosion(&binary_dilation(input, structure), structure)
}

/// Grayscale erosion: min over neighborhood.
pub fn grey_erosion(input: &Array2<f64>, size: usize) -> Array2<f64> {
    minimum_filter(input, size)
}

/// Grayscale dilation: max over neighborhood.
pub fn grey_dilation(input: &Array2<f64>, size: usize) -> Array2<f64> {
    maximum_filter(input, size)
}

/// Connected component labeling with 4-connectivity. Returns (labeled_array, num_features).
pub fn label(input: &Array2<bool>) -> (Array2<usize>, usize) {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut labels = Array2::zeros((rows, cols));
    let mut current_label = 0usize;
    for i in 0..rows {
        for j in 0..cols {
            if input[[i, j]] && labels[[i, j]] == 0 {
                current_label += 1;
                let mut stack = vec![(i, j)];
                while let Some((r, c)) = stack.pop() {
                    if r >= rows || c >= cols || !input[[r, c]] || labels[[r, c]] != 0 {
                        continue;
                    }
                    labels[[r, c]] = current_label;
                    if r > 0 {
                        stack.push((r - 1, c));
                    }
                    if r + 1 < rows {
                        stack.push((r + 1, c));
                    }
                    if c > 0 {
                        stack.push((r, c - 1));
                    }
                    if c + 1 < cols {
                        stack.push((r, c + 1));
                    }
                }
            }
        }
    }
    (labels, current_label)
}

/// Bounding boxes (min_row, max_row, min_col, max_col) for each label.
pub fn find_objects(
    labels: &Array2<usize>,
    num_features: usize,
) -> Vec<Option<(usize, usize, usize, usize)>> {
    let shape = labels.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut result = vec![None; num_features];
    for i in 0..rows {
        for j in 0..cols {
            let lab = labels[[i, j]];
            if lab == 0 {
                continue;
            }
            let idx = lab - 1;
            if idx < num_features {
                match result[idx] {
                    None => {
                        result[idx] = Some((i, i + 1, j, j + 1));
                    }
                    Some((min_r, max_r, min_c, max_c)) => {
                        result[idx] = Some((
                            min_r.min(i),
                            max_r.max(i + 1),
                            min_c.min(j),
                            max_c.max(j + 1),
                        ));
                    }
                }
            }
        }
    }
    result
}

/// Weighted centroid per label.
pub fn center_of_mass(
    input: &Array2<f64>,
    labels: &Array2<usize>,
    num_labels: usize,
) -> Vec<(f64, f64)> {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut sum_val = vec![0.0f64; num_labels];
    let mut sum_r = vec![0.0f64; num_labels];
    let mut sum_c = vec![0.0f64; num_labels];
    for i in 0..rows {
        for j in 0..cols {
            let lab = labels[[i, j]];
            if lab == 0 || lab > num_labels {
                continue;
            }
            let idx = lab - 1;
            let v = input[[i, j]];
            sum_val[idx] += v;
            sum_r[idx] += v * i as f64;
            sum_c[idx] += v * j as f64;
        }
    }
    (0..num_labels)
        .map(|k| {
            if sum_val[k].abs() < 1e-30 {
                (0.0, 0.0)
            } else {
                (sum_r[k] / sum_val[k], sum_c[k] / sum_val[k])
            }
        })
        .collect()
}

/// Sum of input values per label.
pub fn sum_labels(
    input: &Array2<f64>,
    labels: &Array2<usize>,
    num_labels: usize,
) -> Vec<f64> {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut sums = vec![0.0f64; num_labels];
    for i in 0..rows {
        for j in 0..cols {
            let lab = labels[[i, j]];
            if lab == 0 || lab > num_labels {
                continue;
            }
            sums[lab - 1] += input[[i, j]];
        }
    }
    sums
}

/// Mean per label.
pub fn mean_labels(
    input: &Array2<f64>,
    labels: &Array2<usize>,
    num_labels: usize,
) -> Vec<f64> {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut sums = vec![0.0f64; num_labels];
    let mut counts = vec![0usize; num_labels];
    for i in 0..rows {
        for j in 0..cols {
            let lab = labels[[i, j]];
            if lab == 0 || lab > num_labels {
                continue;
            }
            sums[lab - 1] += input[[i, j]];
            counts[lab - 1] += 1;
        }
    }
    (0..num_labels)
        .map(|k| {
            if counts[k] == 0 {
                0.0
            } else {
                sums[k] / counts[k] as f64
            }
        })
        .collect()
}

/// Euclidean distance transform. For each false pixel, distance to nearest true pixel.
pub fn distance_transform_edt(input: &Array2<bool>) -> Array2<f64> {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut dist = Array2::from_elem((rows, cols), f64::INFINITY);
    for i in 0..rows {
        for j in 0..cols {
            if input[[i, j]] {
                dist[[i, j]] = 0.0;
            }
        }
    }
    for i in 0..rows {
        for j in 0..cols {
            if i > 0 {
                let d = dist[[i - 1, j]] + 1.0;
                if d < dist[[i, j]] {
                    dist[[i, j]] = d;
                }
            }
            if j > 0 {
                let d = dist[[i, j - 1]] + 1.0;
                if d < dist[[i, j]] {
                    dist[[i, j]] = d;
                }
            }
        }
    }
    for i in (0..rows).rev() {
        for j in (0..cols).rev() {
            if i + 1 < rows {
                let d = dist[[i + 1, j]] + 1.0;
                if d < dist[[i, j]] {
                    dist[[i, j]] = d;
                }
            }
            if j + 1 < cols {
                let d = dist[[i, j + 1]] + 1.0;
                if d < dist[[i, j]] {
                    dist[[i, j]] = d;
                }
            }
        }
    }
    for i in 0..rows {
        for j in 0..cols {
            dist[[i, j]] = dist[[i, j]].sqrt();
        }
    }
    dist
}

/// 2D convolution with arbitrary kernel.
pub fn convolve(input: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let k_shape = kernel.shape();
    let (k_rows, k_cols) = (k_shape[0], k_shape[1]);
    let k_center_r = k_rows / 2;
    let k_center_c = k_cols / 2;
    let mut output = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for ki in 0..k_rows {
                for kj in 0..k_cols {
                    let ni = i as isize + ki as isize - k_center_r as isize;
                    let nj = j as isize + kj as isize - k_center_c as isize;
                    if ni >= 0
                        && ni < rows as isize
                        && nj >= 0
                        && nj < cols as isize
                    {
                        sum += input[[ni as usize, nj as usize]] * kernel[[ki, kj]];
                    }
                }
            }
            output[[i, j]] = sum;
        }
    }
    output
}

/// 2D cross-correlation (flip kernel, then convolve).
pub fn correlate(input: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let k_shape = kernel.shape();
    let (k_rows, k_cols) = (k_shape[0], k_shape[1]);
    let mut flipped = Array2::zeros((k_rows, k_cols));
    for i in 0..k_rows {
        for j in 0..k_cols {
            flipped[[i, j]] = kernel[[k_rows - 1 - i, k_cols - 1 - j]];
        }
    }
    convolve(input, &flipped)
}

/// Sobel edge magnitude.
pub fn sobel(input: &Array2<f64>) -> Array2<f64> {
    let sx = Array2::from_shape_vec(
        (3, 3),
        vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
    )
    .unwrap();
    let sy = Array2::from_shape_vec(
        (3, 3),
        vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
    )
    .unwrap();
    let gx = convolve(input, &sx);
    let gy = convolve(input, &sy);
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut output = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            output[[i, j]] = (gx[[i, j]].powi(2) + gy[[i, j]].powi(2)).sqrt();
        }
    }
    output
}

/// Prewitt edge magnitude.
pub fn prewitt(input: &Array2<f64>) -> Array2<f64> {
    let px = Array2::from_shape_vec(
        (3, 3),
        vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
    )
    .unwrap();
    let py = Array2::from_shape_vec(
        (3, 3),
        vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let gx = convolve(input, &px);
    let gy = convolve(input, &py);
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut output = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            output[[i, j]] = (gx[[i, j]].powi(2) + gy[[i, j]].powi(2)).sqrt();
        }
    }
    output
}

/// Laplacian filter (second derivative).
pub fn laplace(input: &Array2<f64>) -> Array2<f64> {
    let kernel = Array2::from_shape_vec(
        (3, 3),
        vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    convolve(input, &kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsuniform_filter1d() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = rsuniform_filter1d(&x, 2);
        assert!((y[1] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_filter1d() {
        let input = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let output = gaussian_filter1d(&input, 1.0);
        assert!(output[2] > 0.3);
        assert!(output[1] > 0.0);
        assert!(output[1] > 0.05);
    }

    #[test]
    fn test_gaussian_filter() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let output = gaussian_filter(&input, 1.0);
        assert!(output[[2, 2]] > 0.1);
        assert!(output[[1, 2]] > 0.0);
    }

    #[test]
    fn test_uniform_filter() {
        let input = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 1.0, 1.0, 1.0,
                1.0, 9.0, 9.0, 1.0,
                1.0, 9.0, 9.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let output = uniform_filter(&input, 3);
        assert!(output[[1, 1]] > 1.0);
        assert!(output[[0, 0]] < output[[1, 1]]);
    }

    #[test]
    fn test_median_filter() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 100.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let output = median_filter(&input, 3);
        assert!((output[[1, 1]] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_maximum_filter() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let output = maximum_filter(&input, 3);
        assert!((output[[1, 1]] - 9.0).abs() < 1e-9);
        assert!((output[[0, 0]] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_minimum_filter() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let output = minimum_filter(&input, 3);
        assert!((output[[1, 1]] - 1.0).abs() < 1e-9);
        assert!((output[[0, 0]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_binary_erosion() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, false, false, false,
                false, true, true, true, false,
                false, true, true, true, false,
                false, true, true, true, false,
                false, false, false, false, false,
            ],
        )
        .unwrap();
        let structure = Array2::from_shape_vec(
            (3, 3),
            vec![
                true, true, true,
                true, true, true,
                true, true, true,
            ],
        )
        .unwrap();
        let output = binary_erosion(&input, &structure);
        assert!(output[[2, 2]]);
        assert!(!output[[1, 1]]);
    }

    #[test]
    fn test_binary_dilation() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, true, false, false,
                false, false, false, false, false,
                false, false, false, false, false,
            ],
        )
        .unwrap();
        let structure = Array2::from_shape_vec(
            (3, 3),
            vec![
                true, true, true,
                true, true, true,
                true, true, true,
            ],
        )
        .unwrap();
        let output = binary_dilation(&input, &structure);
        assert!(output[[2, 2]]);
        assert!(output[[1, 2]]);
        assert!(output[[3, 2]]);
    }

    #[test]
    fn test_binary_opening() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, false, false, false,
                false, true, false, false, false,
                false, false, true, true, false,
                false, false, true, true, false,
                false, false, false, false, false,
            ],
        )
        .unwrap();
        let structure = Array2::from_shape_vec(
            (3, 3),
            vec![
                true, true, true,
                true, true, true,
                true, true, true,
            ],
        )
        .unwrap();
        let opened = binary_opening(&input, &structure);
        assert!(!opened[[1, 1]]);
    }

    #[test]
    fn test_binary_closing() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, false, false, false,
                false, true, true, true, false,
                false, true, false, true, false,
                false, true, true, true, false,
                false, false, false, false, false,
            ],
        )
        .unwrap();
        let structure = Array2::from_shape_vec(
            (3, 3),
            vec![
                true, true, true,
                true, true, true,
                true, true, true,
            ],
        )
        .unwrap();
        let closed = binary_closing(&input, &structure);
        assert!(closed[[2, 2]]);
    }

    #[test]
    fn test_grey_erosion() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let output = grey_erosion(&input, 3);
        assert!((output[[1, 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_grey_dilation() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let output = grey_dilation(&input, 3);
        assert!((output[[1, 1]] - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_label() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                true, true, false, false, false,
                true, true, false, false, false,
                false, false, false, true, true,
                false, false, false, true, true,
                false, false, false, false, false,
            ],
        )
        .unwrap();
        let (labels, num) = label(&input);
        assert_eq!(num, 2);
        assert_eq!(labels[[0, 0]], 1);
        assert_eq!(labels[[0, 1]], 1);
        assert_eq!(labels[[2, 3]], 2);
        assert_eq!(labels[[3, 4]], 2);
        assert_eq!(labels[[0, 2]], 0);
    }

    #[test]
    fn test_find_objects() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                true, true, false, false, false,
                true, true, false, false, false,
                false, false, false, true, true,
                false, false, false, true, true,
                false, false, false, false, false,
            ],
        )
        .unwrap();
        let (labels, num) = label(&input);
        let objs = find_objects(&labels, num);
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0], Some((0, 2, 0, 2)));
        assert_eq!(objs[1], Some((2, 4, 3, 5)));
    }

    #[test]
    fn test_center_of_mass() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, 0.0, 0.0,
                0.0, 4.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let labels = Array2::from_shape_vec(
            (3, 3),
            vec![
                0, 0, 0,
                0, 1, 0,
                0, 0, 0,
            ],
        )
        .unwrap();
        let com = center_of_mass(&input, &labels, 1);
        assert!((com[0].0 - 1.0).abs() < 1e-9);
        assert!((com[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sum_labels() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let labels = Array2::from_shape_vec(
            (3, 3),
            vec![
                1, 1, 1,
                1, 1, 1,
                2, 2, 2,
            ],
        )
        .unwrap();
        let sums = sum_labels(&input, &labels, 2);
        assert!((sums[0] - 21.0).abs() < 1e-9);
        assert!((sums[1] - 24.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_labels() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let labels = Array2::from_shape_vec(
            (3, 3),
            vec![
                1, 1, 1,
                1, 1, 1,
                2, 2, 2,
            ],
        )
        .unwrap();
        let means = mean_labels(&input, &labels, 2);
        assert!((means[0] - 3.5).abs() < 1e-9);
        assert!((means[1] - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_distance_transform_edt() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, true, false, false,
                false, false, false, false, false,
                false, false, false, false, false,
            ],
        )
        .unwrap();
        let dist = distance_transform_edt(&input);
        assert!((dist[[2, 2]] - 0.0).abs() < 1e-9);
        assert!((dist[[2, 1]] - 1.0).abs() < 1e-9);
        assert!((dist[[2, 3]] - 1.0).abs() < 1e-9);
        assert!((dist[[1, 2]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_convolve() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let kernel = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let output = convolve(&input, &kernel);
        assert!((output[[1, 1]] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_correlate() {
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let kernel = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let output = correlate(&input, &kernel);
        assert!((output[[1, 1]] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_sobel() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let edges = sobel(&input);
        let total: f64 = edges.sum();
        assert!(total > 0.0, "sobel total={}", total);
    }

    #[test]
    fn test_prewitt() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let edges = prewitt(&input);
        let total: f64 = edges.sum();
        assert!(total > 0.0, "prewitt total={}", total);
    }

    #[test]
    fn test_laplace() {
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let output = laplace(&input);
        assert!(output[[1, 1]] < 0.0);
        assert!(output[[2, 2]].abs() < 1e-9);
    }
}
