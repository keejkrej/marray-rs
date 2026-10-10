use ndarray::{Array2, Array3};

/// Convert RGB image (H, W, 3) to grayscale (H, W) using standard luminosity weights.
pub fn rgb2gray(rgb: &Array3<u8>) -> Array2<u8> {
    let (height, width, channels) = rgb.dim();
    assert_eq!(channels, 3, "Input array must have 3 channels");
    let mut gray = Array2::zeros((height, width));
    for r in 0..height {
        for c in 0..width {
            let r_val = rgb[[r, c, 0]] as f64;
            let g_val = rgb[[r, c, 1]] as f64;
            let b_val = rgb[[r, c, 2]] as f64;
            let gray_val = (0.299 * r_val + 0.587 * g_val + 0.114 * b_val).round() as u8;
            gray[[r, c]] = gray_val;
        }
    }
    gray
}

/// Convert grayscale image (H, W) to RGB (H, W, 3).
pub fn gray2rgb(gray: &Array2<u8>) -> Array3<u8> {
    let (height, width) = gray.dim();
    let mut rgb = Array3::zeros((height, width, 3));
    for r in 0..height {
        for c in 0..width {
            let val = gray[[r, c]];
            rgb[[r, c, 0]] = val;
            rgb[[r, c, 1]] = val;
            rgb[[r, c, 2]] = val;
        }
    }
    rgb
}

/// Convert an RGB image to HSV.
/// Input values in range [0, 1] (or u8/scaled), output H in [0, 1], S in [0, 1], V in [0, 1].
pub fn rgb2hsv(rgb: &Array3<f64>) -> Array3<f64> {
    let (height, width, channels) = rgb.dim();
    assert_eq!(channels, 3, "Input array must have 3 channels");
    let mut hsv = Array3::zeros((height, width, 3));

    for r in 0..height {
        for c in 0..width {
            let red = rgb[[r, c, 0]];
            let green = rgb[[r, c, 1]];
            let blue = rgb[[r, c, 2]];

            let max = red.max(green).max(blue);
            let min = red.min(green).min(blue);
            let delta = max - min;

            // Value
            let v = max;

            // Saturation
            let s = if max > 1e-12 { delta / max } else { 0.0 };

            // Hue
            let mut h = if delta.abs() < 1e-12 {
                0.0
            } else if (max - red).abs() < 1e-12 {
                let segment = (green - blue) / delta;
                let mut h_val = segment % 6.0;
                if h_val < 0.0 { h_val += 6.0; }
                h_val / 6.0
            } else if (max - green).abs() < 1e-12 {
                ((blue - red) / delta + 2.0) / 6.0
            } else {
                ((red - green) / delta + 4.0) / 6.0
            };
            if h < 0.0 { h += 1.0; }
            if h >= 1.0 { h -= 1.0; }

            hsv[[r, c, 0]] = h;
            hsv[[r, c, 1]] = s;
            hsv[[r, c, 2]] = v;
        }
    }
    hsv
}

/// Convert an HSV image to RGB.
/// Input H in [0, 1], S in [0, 1], V in [0, 1], output in [0, 1].
pub fn hsv2rgb(hsv: &Array3<f64>) -> Array3<f64> {
    let (height, width, channels) = hsv.dim();
    assert_eq!(channels, 3, "Input array must have 3 channels");
    let mut rgb = Array3::zeros((height, width, 3));

    for r in 0..height {
        for c in 0..width {
            let h = hsv[[r, c, 0]] * 360.0;
            let s = hsv[[r, c, 1]];
            let v = hsv[[r, c, 2]];

            let c_val = v * s;
            let h_prime = h / 60.0;
            let x = c_val * (1.0 - ((h_prime % 2.0) - 1.0).abs());
            let m = v - c_val;

            let (r1, g1, b1) = if (0.0..1.0).contains(&h_prime) {
                (c_val, x, 0.0)
            } else if (1.0..2.0).contains(&h_prime) {
                (x, c_val, 0.0)
            } else if (2.0..3.0).contains(&h_prime) {
                (0.0, c_val, x)
            } else if (3.0..4.0).contains(&h_prime) {
                (0.0, x, c_val)
            } else if (4.0..5.0).contains(&h_prime) {
                (x, 0.0, c_val)
            } else {
                (c_val, 0.0, x)
            };

            rgb[[r, c, 0]] = r1 + m;
            rgb[[r, c, 1]] = g1 + m;
            rgb[[r, c, 2]] = b1 + m;
        }
    }
    rgb
}
