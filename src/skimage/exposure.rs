use ndarray::Array2;

/// Histogram equalization for contrast enhancement.
pub fn equalize_hist(image: &Array2<u8>) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut hist = [0usize; 256];
    for y in 0..height {
        for x in 0..width {
            hist[image[[y, x]] as usize] += 1;
        }
    }
    let total = height * width;
    let mut cdf = 0usize;
    let mut mapping = [0u8; 256];
    for (i, count) in hist.iter().enumerate() {
        cdf += *count;
        let value = ((cdf as f64 * 255.0) / total as f64).round() as u8;
        mapping[i] = value;
    }

    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = mapping[image[[y, x]] as usize];
        }
    }
    out
}

/// Rescale intensity values from one range to another.
pub fn rescale_intensity(image: &Array2<f64>, in_range: (f64, f64), out_range: (f64, f64)) -> Array2<f64> {
    let (height, width) = image.dim();
    let (in_min, in_max) = in_range;
    let (out_min, out_max) = out_range;
    let input_span = (in_max - in_min).abs();
    let output_span = out_max - out_min;

    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let v = image[[y, x]];
            let scaled = if input_span < 1e-12 {
                0.0
            } else {
                (v - in_min) / input_span
            };
            out[[y, x]] = out_min + scaled * output_span;
        }
    }
    out
}

/// CLAHE (Contrast Limited Adaptive Histogram Equalization).
/// Divides image into tiles, clips histograms, equalizes each tile,
/// and uses bilinear interpolation at tile boundaries.
pub fn equalize_adapthist(image: &Array2<f64>, clip_limit: f64, grid_size: (usize, usize)) -> Array2<f64> {
    let (height, width) = image.dim();
    let (grid_rows, grid_cols) = grid_size;
    let tile_h = height.div_ceil(grid_rows);
    let tile_w = width.div_ceil(grid_cols);

    // find global min/max for normalization
    let (gmin, gmax) = image
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    let grange = if (gmax - gmin).abs() < 1e-15 {
        1.0
    } else {
        gmax - gmin
    };
    let nbins = 256usize;

    // build CDF for each tile
    let mut cdfs: Vec<Vec<f64>> = Vec::with_capacity(grid_rows * grid_cols);
    for tr in 0..grid_rows {
        for tc in 0..grid_cols {
            let r_start = tr * tile_h;
            let c_start = tc * tile_w;
            let r_end = (r_start + tile_h).min(height);
            let c_end = (c_start + tile_w).min(width);

            let mut hist = vec![0.0f64; nbins];
            for y in r_start..r_end {
                for x in c_start..c_end {
                    let val = (image[[y, x]] - gmin) / grange;
                    let bin = ((val * (nbins - 1) as f64).round() as usize).min(nbins - 1);
                    hist[bin] += 1.0;
                }
            }

            // clip histogram
            if clip_limit > 0.0 {
                let excess: f64 = hist.iter().map(|&h| (h - clip_limit).max(0.0)).sum();
                let redistrib = excess / nbins as f64;
                let clip = clip_limit;
                for h in hist.iter_mut() {
                    if *h > clip {
                        *h = clip;
                    }
                    *h += redistrib;
                }
            }

            // build CDF
            let total: f64 = hist.iter().sum();
            let mut cdf = vec![0.0f64; nbins];
            if total > 0.0 {
                let mut cumulative = 0.0;
                for i in 0..nbins {
                    cumulative += hist[i];
                    cdf[i] = cumulative / total;
                }
            }
            cdfs.push(cdf);
        }
    }

    // bilinear interpolation
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            // tile coordinates (floating point)
            let ty = y as f64 / tile_h as f64 - 0.5;
            let tx = x as f64 / tile_w as f64 - 0.5;
            let tr0 = ty.floor().max(0.0) as usize;
            let tc0 = tx.floor().max(0.0) as usize;
            let tr1 = (tr0 + 1).min(grid_rows - 1);
            let tc1 = (tc0 + 1).min(grid_cols - 1);
            let fy = ty - tr0 as f64;
            let fx = tx - tc0 as f64;
            let fy = fy.max(0.0).min(1.0);
            let fx = fx.max(0.0).min(1.0);

            let val = (image[[y, x]] - gmin) / grange;
            let bin = ((val * (nbins - 1) as f64).round() as usize).min(nbins - 1);

            let c00 = cdfs[tr0 * grid_cols + tc0][bin];
            let c01 = cdfs[tr0 * grid_cols + tc1][bin];
            let c10 = cdfs[tr1 * grid_cols + tc0][bin];
            let c11 = cdfs[tr1 * grid_cols + tc1][bin];

            let interp = c00 * (1.0 - fy) * (1.0 - fx)
                + c01 * (1.0 - fy) * fx
                + c10 * fy * (1.0 - fx)
                + c11 * fy * fx;

            out[[y, x]] = interp * grange + gmin;
        }
    }
    out
}

/// Gamma correction: out = in^gamma (normalized to [0,1]).
pub fn adjust_gamma(image: &Array2<f64>, gamma: f64) -> Array2<f64> {
    let (gmin, gmax) = image
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    let range = if (gmax - gmin).abs() < 1e-15 {
        1.0
    } else {
        gmax - gmin
    };
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let norm = ((image[[y, x]] - gmin) / range).max(0.0);
            out[[y, x]] = norm.powf(gamma) * range + gmin;
        }
    }
    out
}

/// Logarithmic adjustment: out = gain * log(1 + in).
pub fn adjust_log(image: &Array2<f64>, gain: f64) -> Array2<f64> {
    let (gmin, gmax) = image
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    let range = if (gmax - gmin).abs() < 1e-15 {
        1.0
    } else {
        gmax - gmin
    };
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let norm = (image[[y, x]] - gmin) / range;
            out[[y, x]] = gain * (1.0 + norm).ln() * range + gmin;
        }
    }
    out
}

/// Histogram of f64 image. Returns (histogram_counts, bin_edges).
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
    fn test_equalize_hist_and_rescale() {
        let image = Array2::from_shape_vec((2, 2), vec![0u8, 255, 0, 255]).unwrap();
        let equalized = equalize_hist(&image);
        assert_eq!(equalized.dim(), (2, 2));
        assert!(equalized.iter().all(|&v| v <= u8::MAX));

        let scaled = rescale_intensity(&Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap(), (0.0, 3.0), (0.0, 1.0));
        assert!((scaled[[0, 0]] - 0.0).abs() < 1e-9);
        assert!((scaled[[1, 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_equalize_adapthist() {
        let mut image = Array2::zeros((8, 8));
        // gradient image
        for y in 0..8 {
            for x in 0..8 {
                image[[y, x]] = (y * 8 + x) as f64 / 63.0;
            }
        }
        let result = equalize_adapthist(&image, 2.0, (2, 2));
        assert_eq!(result.dim(), (8, 8));
        // CLAHE should enhance contrast
        let (min_out, max_out) = result
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });
        assert!(max_out > min_out);
    }

    #[test]
    fn test_equalize_adapthist_flat() {
        let image = Array2::from_elem((8, 8), 0.5);
        let result = equalize_adapthist(&image, 2.0, (4, 4));
        assert_eq!(result.dim(), (8, 8));
        // CLAHE on flat image should produce some output
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_adjust_gamma() {
        let image = Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 0.25, 0.5, 1.0],
        )
        .unwrap();
        let result = adjust_gamma(&image, 2.0);
        assert_eq!(result.dim(), (2, 2));
        // gamma > 1 should darken mid-tones
        assert!(result[[1, 0]] < image[[1, 0]]);
        // endpoints preserved
        assert!((result[[0, 0]] - 0.0).abs() < 1e-9);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adjust_gamma_identity() {
        let image = Array2::from_shape_vec(
            (2, 2),
            vec![0.1, 0.3, 0.5, 0.9],
        )
        .unwrap();
        let result = adjust_gamma(&image, 1.0);
        for (a, b) in image.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_adjust_log() {
        let image = Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 0.25, 0.5, 1.0],
        )
        .unwrap();
        let result = adjust_log(&image, 1.0);
        assert_eq!(result.dim(), (2, 2));
        // log should brighten dark tones relative to bright
        // 0.5 after log(1+0.5) should be less than 0.5
        assert!(result[[1, 0]] < image[[1, 0]] || result[[0, 1]] < image[[0, 1]]);
    }

    #[test]
    fn test_adjust_log_gain() {
        let image = Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 0.25, 0.5, 1.0],
        )
        .unwrap();
        let r1 = adjust_log(&image, 1.0);
        let r2 = adjust_log(&image, 2.0);
        // higher gain = higher output
        assert!(r2[[1, 0]] > r1[[1, 0]]);
    }

    #[test]
    fn test_histogram_exposure() {
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
        // first bin should have 32, last bin should have 32
        assert!((counts[0] - 32.0).abs() < 1e-9);
        assert!((counts[9] - 32.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_uniform() {
        let mut image = Array2::zeros((8, 8));
        for y in 0..8 {
            for x in 0..8 {
                image[[y, x]] = (y * 8 + x) as f64;
            }
        }
        let (counts, _edges) = histogram(&image, 64);
        let total: f64 = counts.iter().sum();
        assert!((total - 64.0).abs() < 1e-9);
    }
}
