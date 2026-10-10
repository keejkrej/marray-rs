use ndarray::Array2;

/// Peak signal-to-noise ratio between two images.
/// Returns f64::INFINITY if images are identical.
pub fn psnr(image1: &Array2<f64>, image2: &Array2<f64>, max_val: f64) -> f64 {
    let mse = mean_squared_error(image1, image2);
    if mse < 1e-15 {
        return f64::INFINITY;
    }
    10.0 * (max_val * max_val / mse).log10()
}

pub use psnr as peak_signal_noise_ratio;

/// Structural similarity index between two images using a sliding window.
/// Returns mean SSIM over all valid window positions.
pub fn structural_similarity(image1: &Array2<f64>, image2: &Array2<f64>, window_size: usize) -> f64 {
    let (height, width) = image1.dim();
    assert_eq!(image1.dim(), image2.dim(), "images must have the same dimensions");
    assert!(window_size <= height && window_size <= width, "window_size exceeds image dimensions");
    assert!(window_size % 2 == 1, "window_size must be odd");

    let l = 1.0;
    let k1 = 0.01f64;
    let k2 = 0.03f64;
    let c1 = (k1 * l).powi(2);
    let c2 = (k2 * l).powi(2);

    let half = window_size / 2;
    let mut ssim_sum = 0.0f64;
    let mut count = 0usize;

    for r in half..height - half {
        for c in half..width - half {
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_xx = 0.0f64;
            let mut sum_yy = 0.0f64;
            let mut sum_xy = 0.0f64;
            let n = (window_size * window_size) as f64;

            for dr in 0..window_size {
                for dc in 0..window_size {
                    let pr = r + dr - half;
                    let pc = c + dc - half;
                    let x = image1[[pr, pc]];
                    let y = image2[[pr, pc]];
                    sum_x += x;
                    sum_y += y;
                    sum_xx += x * x;
                    sum_yy += y * y;
                    sum_xy += x * y;
                }
            }

            let mu_x = sum_x / n;
            let mu_y = sum_y / n;
            let sigma_x2 = sum_xx / n - mu_x * mu_x;
            let sigma_y2 = sum_yy / n - mu_y * mu_y;
            let sigma_xy = sum_xy / n - mu_x * mu_y;

            let num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2);
            let den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2);
            ssim_sum += num / den;
            count += 1;
        }
    }

    if count == 0 {
        return 1.0;
    }
    ssim_sum / count as f64
}

/// Mean squared error between two images.
pub fn mean_squared_error(image1: &Array2<f64>, image2: &Array2<f64>) -> f64 {
    assert_eq!(image1.dim(), image2.dim(), "images must have the same dimensions");
    let n = image1.len() as f64;
    let mut sum = 0.0f64;
    for (&a, &b) in image1.iter().zip(image2.iter()) {
        let d = a - b;
        sum += d * d;
    }
    sum / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_identical() {
        let image = Array2::from_shape_vec((4, 4), vec![0.5; 16]).unwrap();
        let val = psnr(&image, &image, 1.0);
        assert!(val.is_infinite());
    }

    #[test]
    fn test_ssim_identical() {
        let image = Array2::from_shape_vec((8, 8), vec![0.5; 64]).unwrap();
        let val = structural_similarity(&image, &image, 3);
        assert!((val - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mse_known() {
        let a = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 4.0]).unwrap();
        let mse = mean_squared_error(&a, &b);
        assert!((mse - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_psnr_known() {
        let a = Array2::from_shape_vec((2, 2), vec![100.0; 4]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![101.0; 4]).unwrap();
        let val = psnr(&a, &b, 255.0);
        let expected = 10.0 * (255.0f64.powi(2) / 1.0f64).log10();
        assert!((val - expected).abs() < 1e-6);
    }

    #[test]
    fn test_ssim_decreasing() {
        let a = Array2::from_shape_vec((8, 8), vec![0.5; 64]).unwrap();
        let mut b = a.clone();
        b[[3, 3]] = 0.0;
        let s1 = structural_similarity(&a, &a, 3);
        let s2 = structural_similarity(&a, &b, 3);
        assert!(s1 > s2);
    }
}
