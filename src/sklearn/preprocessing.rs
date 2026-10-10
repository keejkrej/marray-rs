use ndarray::{Array1, Array2, Axis};

/// Encode target labels with 0..n_classes-1 values.
pub struct LabelEncoder {
    pub classes: Vec<f64>,
}

impl LabelEncoder {
    pub fn new() -> Self { Self { classes: Vec::new() } }

    pub fn fit(&mut self, y: &Array1<f64>) {
        let mut classes = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes;
    }

    pub fn transform(&self, y: &Array1<f64>) -> Array1<usize> {
        let mut out = Array1::zeros(y.len());
        for (i, value) in y.iter().enumerate() {
            out[i] = self.classes.iter().position(|&c| c == *value).unwrap_or(0);
        }
        out
    }

    pub fn fit_transform(&mut self, y: &Array1<f64>) -> Array1<usize> {
        self.fit(y);
        self.transform(y)
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// StandardScaler standardizes features by removing the mean and scaling to unit variance.
pub struct StandardScaler {
    pub mean: Option<Array1<f64>>,
    pub scale: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: None,
            scale: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let (nrows, ncols) = x.dim();
        if nrows <= 1 {
            let mean = x.mean_axis(Axis(0)).unwrap();
            let scale = Array1::from_elem(ncols, 1.0);
            self.mean = Some(mean);
            self.scale = Some(scale);
            return;
        }
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut std_dev = Array1::zeros(ncols);
        for col in 0..ncols {
            let col_mean = mean[col];
            let mut sum_sq = 0.0;
            for row in 0..nrows {
                sum_sq += (x[[row, col]] - col_mean).powi(2);
            }
            let variance = sum_sq / (nrows as f64);
            let sd = if variance > 1e-14 { variance.sqrt() } else { 1.0 };
            std_dev[col] = sd;
        }
        self.mean = Some(mean);
        self.scale = Some(std_dev);
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().expect("StandardScaler must be fitted before transform");
        let scale = self.scale.as_ref().expect("StandardScaler must be fitted before transform");
        let (nrows, ncols) = x.dim();
        let mut out = Array2::zeros((nrows, ncols));
        for r in 0..nrows {
            for c in 0..ncols {
                out[[r, c]] = (x[[r, c]] - mean[c]) / scale[c];
            }
        }
        out
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x)
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// MinMaxScaler scales features to a given range (typically 0 to 1).
pub struct MinMaxScaler {
    pub min: Option<Array1<f64>>,
    pub max: Option<Array1<f64>>,
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self { min: None, max: None }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let (nrows, ncols) = x.dim();
        if nrows == 0 {
            return;
        }
        let mut mins = Array1::from_elem(ncols, f64::INFINITY);
        let mut maxs = Array1::from_elem(ncols, f64::NEG_INFINITY);
        for r in 0..nrows {
            for c in 0..ncols {
                let v = x[[r, c]];
                if v < mins[c] {
                    mins[c] = v;
                }
                if v > maxs[c] {
                    maxs[c] = v;
                }
            }
        }
        self.min = Some(mins);
        self.max = Some(maxs);
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let min = self.min.as_ref().expect("MinMaxScaler not fitted");
        let max = self.max.as_ref().expect("MinMaxScaler not fitted");
        let (nrows, ncols) = x.dim();
        let mut out = Array2::zeros((nrows, ncols));
        for r in 0..nrows {
            for c in 0..ncols {
                let range = max[c] - min[c];
                let scale = if range > 1e-14 { range } else { 1.0 };
                out[[r, c]] = (x[[r, c]] - min[c]) / scale;
            }
        }
        out
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x)
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// OneHotEncoder encodes categorical features as a one-hot numeric array.
pub struct OneHotEncoder {
    pub categories: Vec<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_label_encoder() {
        let y = array![2.0, 1.0, 2.0, 0.0];
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(&y);
        assert_eq!(encoded.to_vec(), vec![2, 1, 2, 0]);
        assert_eq!(encoder.classes, vec![0.0, 1.0, 2.0]);
    }
}

impl OneHotEncoder {
    /// Create a new OneHotEncoder.
    pub fn new() -> Self {
        Self { categories: Vec::new() }
    }

    /// Fit OneHotEncoder to X.
    pub fn fit(&mut self, x: &Array2<f64>) {
        let (nrows, ncols) = x.dim();
        self.categories = Vec::with_capacity(ncols);
        for col in 0..ncols {
            let mut unique_vals: Vec<f64> = (0..nrows).map(|r| x[[r, col]]).collect();
            unique_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            unique_vals.dedup();
            self.categories.push(unique_vals);
        }
    }

    /// Transform X using one-hot encoding.
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let (nrows, ncols) = x.dim();
        let total_out_cols: usize = self.categories.iter().map(|c| c.len()).sum();
        let mut out = Array2::zeros((nrows, total_out_cols));

        for r in 0..nrows {
            let mut current_col = 0;
            for col in 0..ncols {
                let val = x[[r, col]];
                let cat_list = &self.categories[col];
                let cat_idx = cat_list.iter().position(|&x| x == val).unwrap_or(0);
                out[[r, current_col + cat_idx]] = 1.0;
                current_col += cat_list.len();
            }
        }
        out
    }

    /// Fit to X, then transform it.
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x)
    }
}

