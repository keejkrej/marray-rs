use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// ANOVA F-value between features and classification labels.
pub fn f_classif(x: &Array2<f64>, y: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let (n_samples, n_features) = x.dim();
    assert_eq!(n_samples, y.len());

    // Group samples by class
    let mut classes: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        classes.entry(label.round() as i64).or_default().push(i);
    }
    let k = classes.len();
    assert!(k >= 2, "f_classif requires at least 2 distinct classes");

    let mut f_scores = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for f in 0..n_features {
        let total_mean = x.column(f).mean().unwrap();
        let mut ssb = 0.0; // Between-group sum of squares
        let mut ssw = 0.0; // Within-group sum of squares

        for indices in classes.values() {
            let n_c = indices.len() as f64;
            let group_mean: f64 = indices.iter().map(|&idx| x[[idx, f]]).sum::<f64>() / n_c;
            ssb += n_c * (group_mean - total_mean).powi(2);
            for &idx in indices {
                ssw += (x[[idx, f]] - group_mean).powi(2);
            }
        }

        let df_b = (k - 1) as f64;
        let df_w = (n_samples - k) as f64;
        let msb = ssb / df_b.max(1.0);
        let msw = ssw / df_w.max(1.0);

        let f_stat = if msw > 1e-15 { msb / msw } else { 0.0 };
        // Approximate p-value or store f_stat
        f_scores[f] = f_stat;
        p_values[f] = (-f_stat).exp().min(1.0); // Monotonic surrogate
    }

    (f_scores, p_values)
}

/// Univariate linear regression F-test between features and target.
pub fn f_regression(x: &Array2<f64>, y: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let (n_samples, n_features) = x.dim();
    assert_eq!(n_samples, y.len());

    let y_mean = y.mean().unwrap();
    let y_centered = y - y_mean;
    let ssy: f64 = y_centered.iter().map(|&v| v * v).sum();

    let mut f_scores = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for f in 0..n_features {
        let col = x.column(f);
        let x_mean = col.mean().unwrap();
        let x_centered: Vec<f64> = col.iter().map(|&v| v - x_mean).collect();
        let ssx: f64 = x_centered.iter().map(|&v| v * v).sum();

        if ssx < 1e-15 || ssy < 1e-15 {
            f_scores[f] = 0.0;
            p_values[f] = 1.0;
            continue;
        }

        let cov: f64 = x_centered.iter().zip(y_centered.iter()).map(|(&u, &v)| u * v).sum();
        let r = cov / (ssx * ssy).sqrt();
        let r2 = (r * r).min(1.0 - 1e-12);
        let df = (n_samples - 2) as f64;
        let f_stat = df * r2 / (1.0 - r2);

        f_scores[f] = f_stat;
        p_values[f] = (-f_stat).exp().min(1.0);
    }

    (f_scores, p_values)
}

/// Mutual information between continuous features and discrete class labels.
pub fn mutual_info_classif(x: &Array2<f64>, y: &Array1<f64>, n_bins: Option<usize>) -> Array1<f64> {
    let (n_samples, n_features) = x.dim();
    let bins = n_bins.unwrap_or(10);
    let mut mi_scores = Array1::zeros(n_features);

    // Class probabilities
    let mut class_counts: HashMap<i64, usize> = HashMap::new();
    for &label in y.iter() {
        *class_counts.entry(label.round() as i64).or_insert(0) += 1;
    }
    let n = n_samples as f64;

    for f in 0..n_features {
        let col = x.column(f);
        let min_val = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = (max_val - min_val).max(1e-12);

        let mut joint_counts: HashMap<(usize, i64), usize> = HashMap::new();
        let mut bin_counts = vec![0usize; bins];

        for i in 0..n_samples {
            let val = col[i];
            let bin = (((val - min_val) / range * bins as f64) as usize).min(bins - 1);
            let label = y[i].round() as i64;
            *joint_counts.entry((bin, label)).or_insert(0) += 1;
            bin_counts[bin] += 1;
        }

        let mut mi = 0.0;
        for (&(b_idx, label), &count) in &joint_counts {
            let p_xy = count as f64 / n;
            let p_x = bin_counts[b_idx] as f64 / n;
            let p_y = *class_counts.get(&label).unwrap() as f64 / n;
            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
        mi_scores[f] = mi.max(0.0);
    }

    mi_scores
}

/// Mutual information between continuous features and continuous target.
pub fn mutual_info_reg(x: &Array2<f64>, y: &Array1<f64>, n_bins: Option<usize>) -> Array1<f64> {
    let (n_samples, n_features) = x.dim();
    let bins = n_bins.unwrap_or(10);
    let mut mi_scores = Array1::zeros(n_features);
    let n = n_samples as f64;

    let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_range = (y_max - y_min).max(1e-12);

    let mut y_bins = vec![0usize; n_samples];
    let mut y_bin_counts = vec![0usize; bins];
    for i in 0..n_samples {
        let b = (((y[i] - y_min) / y_range * bins as f64) as usize).min(bins - 1);
        y_bins[i] = b;
        y_bin_counts[b] += 1;
    }

    for f in 0..n_features {
        let col = x.column(f);
        let x_min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let x_range = (x_max - x_min).max(1e-12);

        let mut joint_counts = HashMap::new();
        let mut x_bin_counts = vec![0usize; bins];

        for i in 0..n_samples {
            let bx = (((col[i] - x_min) / x_range * bins as f64) as usize).min(bins - 1);
            let by = y_bins[i];
            *joint_counts.entry((bx, by)).or_insert(0) += 1;
            x_bin_counts[bx] += 1;
        }

        let mut mi = 0.0;
        for (&(bx, by), &count) in &joint_counts {
            let p_xy = count as f64 / n;
            let p_x = x_bin_counts[bx] as f64 / n;
            let p_y = y_bin_counts[by] as f64 / n;
            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
        mi_scores[f] = mi.max(0.0);
    }

    mi_scores
}

/// Select features according to the k highest scores.
pub struct SelectKBest {
    pub k: usize,
    pub scores: Option<Array1<f64>>,
    pub selected_features: Option<Vec<usize>>,
}

impl SelectKBest {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            scores: None,
            selected_features: None,
        }
    }

    pub fn fit_with_scores(&mut self, scores: Array1<f64>) -> Result<(), String> {
        let n_features = scores.len();
        let k = self.k.min(n_features);

        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        indices.truncate(k);
        indices.sort(); // keep original relative order
        self.selected_features = Some(indices);
        self.scores = Some(scores);
        Ok(())
    }

    pub fn fit_classif(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (f_scores, _) = f_classif(x, y);
        self.fit_with_scores(f_scores)
    }

    pub fn fit_regression(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (f_scores, _) = f_regression(x, y);
        self.fit_with_scores(f_scores)
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let selected = self.selected_features.as_ref().expect("Model not fitted");
        let (nrows, _) = x.dim();
        let mut out = Array2::zeros((nrows, selected.len()));
        for (new_col, &orig_col) in selected.iter().enumerate() {
            for r in 0..nrows {
                out[[r, new_col]] = x[[r, orig_col]];
            }
        }
        out
    }

    pub fn fit_transform_classif(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.fit_classif(x, y)?;
        Ok(self.transform(x))
    }

    pub fn fit_transform_regression(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.fit_regression(x, y)?;
        Ok(self.transform(x))
    }
}

/// Recursive Feature Elimination (RFE).
pub struct RFE {
    pub n_features_to_select: usize,
    pub step: usize,
    pub ranking: Option<Array1<usize>>,
    pub support: Option<Vec<bool>>,
}

impl RFE {
    pub fn new(n_features_to_select: usize, step: Option<usize>) -> Self {
        Self {
            n_features_to_select,
            step: step.unwrap_or(1),
            ranking: None,
            support: None,
        }
    }

    /// Fit RFE using univariate linear correlations / weights as the feature importance metric.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (n_samples, n_features) = x.dim();
        let target_k = self.n_features_to_select.min(n_features);

        let mut active: Vec<usize> = (0..n_features).collect();
        let mut ranking = vec![1usize; n_features];
        let mut current_rank = n_features - target_k + 1;

        while active.len() > target_k {
            let cur_k = active.len();
            let mut sub_x = Array2::zeros((n_samples, cur_k));
            for (col_idx, &feat_idx) in active.iter().enumerate() {
                for r in 0..n_samples {
                    sub_x[[r, col_idx]] = x[[r, feat_idx]];
                }
            }

            let (scores, _) = f_regression(&sub_x, y);

            // Find feature with lowest score in active set
            let mut score_pairs: Vec<(usize, f64)> = active
                .iter()
                .enumerate()
                .map(|(i, &feat)| (feat, scores[i]))
                .collect();
            score_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let n_remove = self.step.min(active.len() - target_k);
            for i in 0..n_remove {
                let feat_to_remove = score_pairs[i].0;
                ranking[feat_to_remove] = current_rank;
                active.retain(|&f| f != feat_to_remove);
            }
            current_rank = current_rank.saturating_sub(n_remove).max(2);
        }

        let mut support = vec![false; n_features];
        for &feat in &active {
            support[feat] = true;
            ranking[feat] = 1;
        }

        self.ranking = Some(Array1::from_vec(ranking));
        self.support = Some(support);
        Ok(())
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let support = self.support.as_ref().expect("RFE not fitted");
        let selected: Vec<usize> = support
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i)
            .collect();
        let (nrows, _) = x.dim();
        let mut out = Array2::zeros((nrows, selected.len()));
        for (new_col, &orig_col) in selected.iter().enumerate() {
            for r in 0..nrows {
                out[[r, new_col]] = x[[r, orig_col]];
            }
        }
        out
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.fit(x, y)?;
        Ok(self.transform(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_f_classif_and_select_k_best() {
        // Feature 0 separates classes perfectly, feature 1 is random noise
        let x = array![
            [1.0, 5.0],
            [1.2, 5.2],
            [9.0, 5.1],
            [9.2, 4.9]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut selector = SelectKBest::new(1);
        let x_trans = selector.fit_transform_classif(&x, &y).unwrap();
        assert_eq!(x_trans.dim(), (4, 1));
        assert_eq!(selector.selected_features.unwrap(), vec![0]);
    }

    #[test]
    fn test_mutual_info() {
        let x = array![
            [1.0, 10.0],
            [2.0, 10.0],
            [10.0, 1.0],
            [11.0, 1.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let mi = mutual_info_classif(&x, &y, Some(3));
        assert!(mi[0] > 0.0);
    }

    #[test]
    fn test_rfe() {
        let x = array![
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 0.1],
            [3.0, 30.0, 0.0],
            [4.0, 40.0, 0.1]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let mut rfe = RFE::new(2, Some(1));
        let x_sub = rfe.fit_transform(&x, &y).unwrap();
        assert_eq!(x_sub.dim(), (4, 2));
    }
}
