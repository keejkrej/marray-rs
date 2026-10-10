use ndarray::Array2;
use image::{GrayImage, Luma};

/// Compute Otsu's optimal threshold for a grayscale image.
pub fn threshold_otsu(image: &Array2<u8>) -> f64 {
    let (height, width) = image.dim();
    let total = height * width;
    let mut hist = [0usize; 256];
    for y in 0..height {
        for x in 0..width {
            hist[image[[y, x]] as usize] += 1;
        }
    }
    let mut sum = 0.0;
    for i in 0..256 {
        sum += i as f64 * hist[i] as f64;
    }
    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    let mut max_var = 0.0;
    let mut threshold = 0.0;
    for i in 0..256 {
        w_b += hist[i] as f64;
        if w_b <= 0.0 || w_b >= total as f64 { continue; }
        let w_f = total as f64 - w_b;
        if w_f <= 0.0 { break; }
        sum_b += i as f64 * hist[i] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var_between > max_var {
            max_var = var_between;
            threshold = i as f64;
        }
    }
    threshold
}

/// Apply Gaussian blur to a grayscale image.
pub fn gaussian(image: &Array2<u8>, sigma: f64) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut gray_img = GrayImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            gray_img.put_pixel(x as u32, y as u32, Luma([image[[y, x]]]));
        }
    }

    let blurred = image::imageops::blur(&gray_img, sigma as f32);

    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = blurred.get_pixel(x as u32, y as u32)[0];
        }
    }
    out
}

/// Apply Sobel filter to find edges on a grayscale image.
pub fn sobel(image: &Array2<u8>) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut gray_img = GrayImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            gray_img.put_pixel(x as u32, y as u32, Luma([image[[y, x]]]));
        }
    }

    let grad = imageproc::gradients::sobel_gradients(&gray_img);

    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = grad.get_pixel(x as u32, y as u32)[0].min(255) as u8;
        }
    }
    out
}

/// Apply Canny edge detection filter to a grayscale image.
pub fn canny(image: &Array2<u8>, low_threshold: f32, high_threshold: f32) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut gray_img = GrayImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            gray_img.put_pixel(x as u32, y as u32, Luma([image[[y, x]]]));
        }
    }

    let edges = imageproc::edges::canny(&gray_img, low_threshold, high_threshold);

    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = edges.get_pixel(x as u32, y as u32)[0];
        }
    }
    out
}

/// Laplacian filter using kernel [[0,1,0],[1,-4,1],[0,1,0]].
pub fn laplace(image: &Array2<f64>) -> Array2<f64> {
    let kernel =
        Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0])
            .unwrap();
    convolve_2d(image, &kernel)
}

/// Median filter with square window of given size.
pub fn median(image: &Array2<f64>, size: usize) -> Array2<f64> {
    let (height, width) = image.dim();
    let half = size / 2;
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut vals = Vec::new();
            for ny in y.saturating_sub(half)..=(y + half).min(height - 1) {
                for nx in x.saturating_sub(half)..=(x + half).min(width - 1) {
                    vals.push(image[[ny, nx]]);
                }
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out[[y, x]] = vals[vals.len() / 2];
        }
    }
    out
}

/// Local adaptive threshold. Method can be "mean" or "gaussian".
pub fn threshold_local(image: &Array2<f64>, block_size: usize, method: &str) -> Array2<f64> {
    let (height, width) = image.dim();
    let blurred = match method {
        "gaussian" => {
            let sigma = block_size as f64 / 6.0;
            gaussian_filter_f64(image, sigma)
        }
        _ => mean_filter(image, block_size),
    };
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = image[[y, x]] - blurred[[y, x]];
        }
    }
    out
}

/// Yen's threshold method.
pub fn threshold_yen(image: &Array2<f64>) -> f64 {
    let (counts, _edges) = histogram(image, 256);
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut prob = vec![0.0f64; 256];
    for i in 0..256 {
        prob[i] = counts[i] / total;
    }
    // cumulative sums
    let mut cdf = vec![0.0f64; 256];
    cdf[0] = prob[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + prob[i];
    }
    let mut max_criterion = f64::NEG_INFINITY;
    let mut best_threshold = 0usize;
    for t in 1..256 {
        let p0 = cdf[t - 1];
        let p1 = 1.0 - p0;
        if p0 <= 0.0 || p1 <= 0.0 {
            continue;
        }
        // entropy of background
        let mut h0 = 0.0f64;
        for i in 0..t {
            if prob[i] > 0.0 {
                let p = prob[i] / p0;
                h0 -= p * p.ln();
            }
        }
        // entropy of foreground
        let mut h1 = 0.0f64;
        for i in t..256 {
            if prob[i] > 0.0 {
                let p = prob[i] / p1;
                h1 -= p * p.ln();
            }
        }
        let criterion = -2.0 * (p0 * h0 + p1 * h1).ln() + 2.0 * (p0 * p0 * h0 + p1 * p1 * h1).ln();
        if criterion.is_finite() && criterion > max_criterion {
            max_criterion = criterion;
            best_threshold = t;
        }
    }
    best_threshold as f64
}

/// Li's minimum cross-entropy threshold.
pub fn threshold_li(image: &Array2<f64>) -> f64 {
    let (counts, _edges) = histogram(image, 256);
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut threshold = {
        let mut sum = 0.0;
        for i in 0..256 {
            sum += i as f64 * counts[i];
        }
        (sum / total).round() as usize
    };
    threshold = threshold.min(254).max(1);
    for _ in 0..1000 {
        let (mut t_back, mut t_fore) = (0.0f64, 0.0f64);
        let (mut w_back, mut w_fore) = (0.0f64, 0.0f64);
        for i in 0..=threshold {
            t_back += i as f64 * counts[i];
            w_back += counts[i];
        }
        for i in (threshold + 1)..256 {
            t_fore += i as f64 * counts[i];
            w_fore += counts[i];
        }
        if w_back == 0.0 || w_fore == 0.0 {
            break;
        }
        let mean_back = t_back / w_back;
        let mean_fore = t_fore / w_fore;
        let new_t = ((mean_back + mean_fore) / 2.0).round() as usize;
        let new_t = new_t.min(254).max(1);
        if new_t == threshold {
            break;
        }
        threshold = new_t;
    }
    threshold as f64
}

/// Triangle method threshold.
pub fn threshold_triangle(image: &Array2<f64>) -> f64 {
    let (counts, _edges) = histogram(image, 256);
    // find min and max bin with nonzero count
    let mut first = 0usize;
    let mut last = 0usize;
    for i in 0..256 {
        if counts[i] > 0.0 {
            first = i;
            break;
        }
    }
    for i in (0..256).rev() {
        if counts[i] > 0.0 {
            last = i;
            break;
        }
    }
    if first >= last {
        return first as f64;
    }
    // find peak bin
    let mut peak = first;
    for i in first..=last {
        if counts[i] > counts[peak] {
            peak = i;
        }
    }
    // line from (peak, counts[peak]) to (last, 0)
    let dx = last as f64 - peak as f64;
    let dy = 0.0 - counts[peak];
    let line_len = (dx * dx + dy * dy).sqrt();
    if line_len < 1e-12 {
        return peak as f64;
    }
    let mut max_dist = 0.0f64;
    let mut best = first;
    for i in first..=last {
        // distance from point (i, counts[i]) to line
        let px = i as f64 - peak as f64;
        let py = counts[i] - counts[peak];
        let dist = (dy * px - dx * py).abs() / line_len;
        if dist > max_dist {
            max_dist = dist;
            best = i;
        }
    }
    best as f64
}

/// Unsharp masking: image + amount * (image - blurred).
pub fn unsharp_mask(image: &Array2<f64>, radius: f64, amount: f64) -> Array2<f64> {
    let blurred = gaussian_filter_f64(image, radius);
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = image[[y, x]] + amount * (image[[y, x]] - blurred[[y, x]]);
        }
    }
    out
}

/// Prewitt edge detection (f64 version).
pub fn prewitt(image: &Array2<f64>) -> Array2<f64> {
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
    let gx = convolve_2d(image, &px);
    let gy = convolve_2d(image, &py);
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = (gx[[y, x]].powi(2) + gy[[y, x]].powi(2)).sqrt();
        }
    }
    out
}

/// Scharr edge detection.
pub fn scharr(image: &Array2<f64>) -> Array2<f64> {
    let sx = Array2::from_shape_vec(
        (3, 3),
        vec![-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0],
    )
    .unwrap();
    let sy = Array2::from_shape_vec(
        (3, 3),
        vec![-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0],
    )
    .unwrap();
    let gx = convolve_2d(image, &sx);
    let gy = convolve_2d(image, &sy);
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = (gx[[y, x]].powi(2) + gy[[y, x]].powi(2)).sqrt();
        }
    }
    out
}

/// Difference of Gaussians filter.
pub fn difference_of_gaussians(image: &Array2<f64>, sigma1: f64, sigma2: f64) -> Array2<f64> {
    let g1 = gaussian_filter_f64(image, sigma1);
    let g2 = gaussian_filter_f64(image, sigma2);
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = g1[[y, x]] - g2[[y, x]];
        }
    }
    out
}

/// Internal: 2D convolution for f64 arrays.
fn convolve_2d(image: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let (height, width) = image.dim();
    let (kh, kw) = kernel.dim();
    let kcy = kh / 2;
    let kcx = kw / 2;
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for ky in 0..kh {
                for kx in 0..kw {
                    let ny = y as isize + ky as isize - kcy as isize;
                    let nx = x as isize + kx as isize - kcx as isize;
                    if ny >= 0 && ny < height as isize && nx >= 0 && nx < width as isize {
                        sum += image[[ny as usize, nx as usize]] * kernel[[ky, kx]];
                    }
                }
            }
            out[[y, x]] = sum;
        }
    }
    out
}

/// Internal: mean filter for threshold_local.
fn mean_filter(image: &Array2<f64>, size: usize) -> Array2<f64> {
    let (height, width) = image.dim();
    let half = size / 2;
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count = 0usize;
            for ny in y.saturating_sub(half)..=(y + half).min(height - 1) {
                for nx in x.saturating_sub(half)..=(x + half).min(width - 1) {
                    sum += image[[ny, nx]];
                    count += 1;
                }
            }
            out[[y, x]] = sum / count as f64;
        }
    }
    out
}

/// Internal: 2D Gaussian filter for f64 arrays (separable).
fn gaussian_filter_f64(image: &Array2<f64>, sigma: f64) -> Array2<f64> {
    let (height, width) = image.dim();
    let radius = (3.0 * sigma).ceil() as usize;
    let ksize = 2 * radius + 1;
    let mut kernel = vec![0.0f64; ksize];
    let mut sum = 0.0;
    for i in 0..ksize {
        let x = i as f64 - radius as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }
    // horizontal pass
    let mut temp = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut val = 0.0;
            for k in 0..ksize {
                let nx = x as isize + k as isize - radius as isize;
                let nx = if nx < 0 {
                    (-nx) as usize
                } else if nx >= width as isize {
                    2 * width - 1 - nx as usize
                } else {
                    nx as usize
                };
                val += image[[y, nx]] * kernel[k];
            }
            temp[[y, x]] = val;
        }
    }
    // vertical pass
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut val = 0.0;
            for k in 0..ksize {
                let ny = y as isize + k as isize - radius as isize;
                let ny = if ny < 0 {
                    (-ny) as usize
                } else if ny >= height as isize {
                    2 * height - 1 - ny as usize
                } else {
                    ny as usize
                };
                val += temp[[ny, x]] * kernel[k];
            }
            out[[y, x]] = val;
        }
    }
    out
}

/// Histogram of f64 image. Returns (counts, bin_edges).
pub fn histogram(image: &Array2<f64>, nbins: usize) -> (Vec<f64>, Vec<f64>) {
    let (min_val, max_val) = image
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    let mut counts = vec![0.0f64; nbins];
    let mut edges = Vec::with_capacity(nbins + 1);
    let range = max_val - min_val;
    if range < 1e-15 {
        edges.push(min_val);
        for _ in 0..nbins {
            edges.push(max_val);
        }
        counts[0] = image.len() as f64;
        return (counts, edges);
    }
    let bin_width = range / nbins as f64;
    for i in 0..=nbins {
        edges.push(min_val + i as f64 * bin_width);
    }
    for &v in image.iter() {
        let mut bin = ((v - min_val) / bin_width) as usize;
        if bin >= nbins {
            bin = nbins - 1;
        }
        counts[bin] += 1.0;
    }
    (counts, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace() {
        // flat image -> laplacian should be zero
        let image = Array2::from_elem((4, 4), 5.0);
        let lap = laplace(&image);
        for y in 1..3 {
            for x in 1..3 {
                assert!(lap[[y, x]].abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_laplace_edge() {
        // step edge: left half 0, right half 1
        let mut image = Array2::zeros((8, 8));
        for y in 0..8 {
            for x in 4..8 {
                image[[y, x]] = 1.0;
            }
        }
        let lap = laplace(&image);
        // at the boundary, laplacian should be nonzero
        assert!(lap[[4, 4]].abs() > 0.01);
    }

    #[test]
    fn test_median() {
        let image = Array2::from_shape_vec(
            (3, 3),
            vec![9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let result = median(&image, 3);
        // median of 9 ones and one 9 -> should be 1
        assert!((result[[0, 0]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_median_salt_pepper() {
        let mut image = Array2::from_elem((8, 8), 0.5);
        // add salt and pepper noise
        image[[0, 0]] = 0.0;
        image[[1, 1]] = 1.0;
        image[[2, 2]] = 0.0;
        let result = median(&image, 3);
        // median should remove noise
        assert!((result[[1, 1]] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_threshold_local_mean() {
        let mut image = Array2::zeros((8, 8));
        for y in 0..8 {
            for x in 0..8 {
                image[[y, x]] = (y * 8 + x) as f64 / 63.0;
            }
        }
        let result = threshold_local(&image, 3, "mean");
        assert_eq!(result.dim(), (8, 8));
        // result should have both positive and negative values
        let has_pos = result.iter().any(|&v| v > 0.0);
        let has_neg = result.iter().any(|&v| v < 0.0);
        assert!(has_pos && has_neg);
    }

    #[test]
    fn test_threshold_local_gaussian() {
        let mut image = Array2::zeros((8, 8));
        for y in 0..8 {
            for x in 0..8 {
                image[[y, x]] = (y * 8 + x) as f64;
            }
        }
        let result = threshold_local(&image, 3, "gaussian");
        assert_eq!(result.dim(), (8, 8));
    }

    #[test]
    fn test_threshold_yen() {
        // bimodal image
        let mut image = Array2::zeros((8, 8));
        for y in 0..4 {
            for x in 0..8 {
                image[[y, x]] = 0.1;
            }
        }
        for y in 4..8 {
            for x in 0..8 {
                image[[y, x]] = 0.9;
            }
        }
        let t = threshold_yen(&image);
        assert!(t >= 0.0 && t <= 1.0);
    }

    #[test]
    fn test_threshold_li() {
        let mut image = Array2::zeros((8, 8));
        for y in 0..4 {
            for x in 0..8 {
                image[[y, x]] = 0.2;
            }
        }
        for y in 4..8 {
            for x in 0..8 {
                image[[y, x]] = 0.8;
            }
        }
        let t = threshold_li(&image);
        assert!(t > 0.0);
    }

    #[test]
    fn test_threshold_triangle() {
        let mut image = Array2::zeros((8, 8));
        // mostly dark with a few bright pixels
        for y in 0..8 {
            for x in 0..8 {
                image[[y, x]] = 0.1;
            }
        }
        image[[7, 7]] = 1.0;
        let t = threshold_triangle(&image);
        assert!(t >= 0.0);
    }

    #[test]
    fn test_unsharp_mask() {
        let image = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let sharpened = unsharp_mask(&image, 1.0, 1.5);
        assert_eq!(sharpened.dim(), (4, 4));
        // sharpened should have higher contrast at edges
        assert!(sharpened[[1, 1]] > image[[1, 1]] || sharpened[[1, 0]] < image[[1, 0]]);
    }

    #[test]
    fn test_prewitt() {
        let mut image = Array2::zeros((8, 8));
        for y in 0..8 {
            for x in 4..8 {
                image[[y, x]] = 1.0;
            }
        }
        let edges = prewitt(&image);
        // should detect vertical edge at x=4
        assert!(edges[[4, 3]] > 0.1 || edges[[4, 4]] > 0.1);
    }

    #[test]
    fn test_scharr() {
        let mut image = Array2::zeros((8, 8));
        for y in 0..8 {
            for x in 4..8 {
                image[[y, x]] = 1.0;
            }
        }
        let edges = scharr(&image);
        assert!(edges[[4, 3]] > 0.1 || edges[[4, 4]] > 0.1);
    }

    #[test]
    fn test_difference_of_gaussians() {
        let mut image = Array2::zeros((8, 8));
        image[[4, 4]] = 1.0;
        let dog = difference_of_gaussians(&image, 1.0, 2.0);
        assert_eq!(dog.dim(), (8, 8));
        // center should have a strong response
        assert!(dog[[4, 4]].abs() > 0.01);
    }

    #[test]
    fn test_histogram() {
        let mut image = Array2::zeros((8, 8));
        for y in 0..4 {
            for x in 0..8 {
                image[[y, x]] = 0.0;
            }
        }
        for y in 4..8 {
            for x in 0..8 {
                image[[y, x]] = 1.0;
            }
        }
        let (counts, edges) = histogram(&image, 10);
        assert_eq!(counts.len(), 10);
        assert_eq!(edges.len(), 11);
        let total: f64 = counts.iter().sum();
        assert!((total - 64.0).abs() < 1e-9);
    }
}

