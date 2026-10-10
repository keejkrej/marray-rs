use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Linear Discriminant Analysis (LDA) classifier and dimensionality reducer.
pub struct LinearDiscriminantAnalysis {
    pub n_components: Option<usize>,
    pub classes: Option<Vec<f64>>,
    pub priors: Option<Array1<f64>>,
    pub means: Option<Array2<f64>>, // (n_classes, n_features)
    pub covariance: Option<Array2<f64>>,
    pub scalings: Option<Array2<f64>>, // projection matrix for transform
    cov_inv: Option<Array2<f64>>,
}

impl LinearDiscriminantAnalysis {
    pub fn new(n_components: Option<usize>) -> Self {
        Self {
            n_components,
            classes: None,
            priors: None,
            means: None,
            covariance: None,
            scalings: None,
            cov_inv: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (n_samples, n_features) = x.dim();
        assert_eq!(n_samples, y.len());

        let mut class_map: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_map.entry(label.round() as i64).or_default().push(i);
        }

        let mut unique_labels: Vec<i64> = class_map.keys().cloned().collect();
        unique_labels.sort();
        let n_classes = unique_labels.len();
        if n_classes < 2 {
            return Err("LDA requires at least 2 classes".to_string());
        }

        let mut classes = Vec::with_capacity(n_classes);
        let mut priors = Array1::zeros(n_classes);
        let mut means = Array2::zeros((n_classes, n_features));
        let mut pooled_cov = Array2::zeros((n_features, n_features));

        let overall_mean = x.mean_axis(ndarray::Axis(0)).unwrap();

        for (c_idx, &label) in unique_labels.iter().enumerate() {
            classes.push(label as f64);
            let indices = &class_map[&label];
            let n_c = indices.len() as f64;
            priors[c_idx] = n_c / n_samples as f64;

            for &i in indices {
                for f in 0..n_features {
                    means[[c_idx, f]] += x[[i, f]];
                }
            }
            for f in 0..n_features {
                means[[c_idx, f]] /= n_c;
            }

            for &i in indices {
                for j in 0..n_features {
                    let diff_j = x[[i, j]] - means[[c_idx, j]];
                    for k in 0..n_features {
                        let diff_k = x[[i, k]] - means[[c_idx, k]];
                        pooled_cov[[j, k]] += diff_j * diff_k;
                    }
                }
            }
        }

        let denom = (n_samples - n_classes).max(1) as f64;
        pooled_cov.mapv_inplace(|v| v / denom);

        // Add regularizer on diagonal for numerical stability
        for f in 0..n_features {
            pooled_cov[[f, f]] += 1e-6;
        }

        let cov_inv = crate::sp::linalg::inv(&pooled_cov)
            .map_err(|_| "Failed to invert pooled covariance matrix".to_string())?;

        // Dimensionality reduction scalings (between-class scatter S_b)
        let mut s_b = Array2::zeros((n_features, n_features));
        for (c_idx, &label) in unique_labels.iter().enumerate() {
            let n_c = class_map[&label].len() as f64;
            let m_diff = means.row(c_idx).to_owned() - &overall_mean;
            for j in 0..n_features {
                for k in 0..n_features {
                    s_b[[j, k]] += n_c * m_diff[j] * m_diff[k];
                }
            }
        }

        let max_k = (n_classes - 1).min(n_features);
        let k = self.n_components.unwrap_or(max_k).min(max_k);

        let sw_inv_sb = cov_inv.dot(&s_b);
        let svd = crate::sp::linalg::svd(&sw_inv_sb);
        let vh = svd.vh.unwrap_or_else(|| Array2::eye(n_features));

        let mut scalings = Array2::zeros((n_features, k));
        for f in 0..n_features {
            for comp in 0..k {
                scalings[[f, comp]] = vh[[comp, f]];
            }
        }

        self.classes = Some(classes);
        self.priors = Some(priors);
        self.means = Some(means);
        self.covariance = Some(pooled_cov);
        self.cov_inv = Some(cov_inv);
        self.scalings = Some(scalings);
        Ok(())
    }

    /// Linear discriminant function: delta_c(x) = x^T Sigma^{-1} mu_c - 0.5 mu_c^T Sigma^{-1} mu_c + ln(pi_c)
    pub fn decision_function(&self, x: &Array2<f64>) -> Array2<f64> {
        let (n_samples, _) = x.dim();
        let means = self.means.as_ref().expect("Model not fitted");
        let priors = self.priors.as_ref().expect("Model not fitted");
        let cov_inv = self.cov_inv.as_ref().expect("Model not fitted");
        let n_classes = priors.len();

        let mut df = Array2::zeros((n_samples, n_classes));

        for c in 0..n_classes {
            let mu_c = means.row(c).to_owned();
            let inv_mu = cov_inv.dot(&mu_c);
            let const_term = -0.5 * mu_c.dot(&inv_mu) + priors[c].ln();

            for i in 0..n_samples {
                let xi = x.row(i);
                df[[i, c]] = xi.dot(&inv_mu) + const_term;
            }
        }

        df
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let df = self.decision_function(x);
        let classes = self.classes.as_ref().expect("Model not fitted");
        let (n_samples, n_classes) = df.dim();
        let mut preds = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_c = 0;
            for c in 0..n_classes {
                if df[[i, c]] > best_val {
                    best_val = df[[i, c]];
                    best_c = c;
                }
            }
            preds[i] = classes[best_c];
        }

        preds
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let scalings = self.scalings.as_ref().expect("Model not fitted");
        x.dot(scalings)
    }
}

/// Quadratic Discriminant Analysis (QDA) classifier.
pub struct QuadraticDiscriminantAnalysis {
    pub classes: Option<Vec<f64>>,
    pub priors: Option<Array1<f64>>,
    pub means: Option<Array2<f64>>,
    pub cov_invs: Option<Vec<Array2<f64>>>,
    pub log_dets: Option<Vec<f64>>,
}

impl QuadraticDiscriminantAnalysis {
    pub fn new() -> Self {
        Self {
            classes: None,
            priors: None,
            means: None,
            cov_invs: None,
            log_dets: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (n_samples, n_features) = x.dim();
        let mut class_map: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_map.entry(label.round() as i64).or_default().push(i);
        }

        let mut unique_labels: Vec<i64> = class_map.keys().cloned().collect();
        unique_labels.sort();
        let n_classes = unique_labels.len();

        let mut classes = Vec::with_capacity(n_classes);
        let mut priors = Array1::zeros(n_classes);
        let mut means = Array2::zeros((n_classes, n_features));
        let mut cov_invs = Vec::with_capacity(n_classes);
        let mut log_dets = Vec::with_capacity(n_classes);

        for (c_idx, &label) in unique_labels.iter().enumerate() {
            classes.push(label as f64);
            let indices = &class_map[&label];
            let n_c = indices.len() as f64;
            priors[c_idx] = n_c / n_samples as f64;

            for &i in indices {
                for f in 0..n_features {
                    means[[c_idx, f]] += x[[i, f]];
                }
            }
            for f in 0..n_features {
                means[[c_idx, f]] /= n_c;
            }

            let mut class_cov = Array2::zeros((n_features, n_features));
            for &i in indices {
                for j in 0..n_features {
                    let dj = x[[i, j]] - means[[c_idx, j]];
                    for k in 0..n_features {
                        let dk = x[[i, k]] - means[[c_idx, k]];
                        class_cov[[j, k]] += dj * dk;
                    }
                }
            }

            let denom = (n_c - 1.0).max(1.0);
            class_cov.mapv_inplace(|v| v / denom);
            for f in 0..n_features {
                class_cov[[f, f]] += 1e-5;
            }

            let inv = crate::sp::linalg::inv(&class_cov)
                .map_err(|_| "Failed to invert class covariance".to_string())?;
            let det = crate::sp::linalg::det(&class_cov);
            let log_det = if det > 0.0 { det.ln() } else { 0.0 };

            cov_invs.push(inv);
            log_dets.push(log_det);
        }

        self.classes = Some(classes);
        self.priors = Some(priors);
        self.means = Some(means);
        self.cov_invs = Some(cov_invs);
        self.log_dets = Some(log_dets);
        Ok(())
    }

    /// delta_c(x) = -0.5 * ln|Sigma_c| - 0.5 * (x - mu_c)^T Sigma_c^{-1} (x - mu_c) + ln(pi_c)
    pub fn decision_function(&self, x: &Array2<f64>) -> Array2<f64> {
        let (n_samples, n_features) = x.dim();
        let means = self.means.as_ref().expect("Model not fitted");
        let priors = self.priors.as_ref().expect("Model not fitted");
        let cov_invs = self.cov_invs.as_ref().expect("Model not fitted");
        let log_dets = self.log_dets.as_ref().expect("Model not fitted");
        let n_classes = priors.len();

        let mut df = Array2::zeros((n_samples, n_classes));

        for c in 0..n_classes {
            let mu_c = means.row(c);
            let inv = &cov_invs[c];
            let log_det = log_dets[c];
            let log_prior = priors[c].ln();

            for i in 0..n_samples {
                let mut diff = Array1::zeros(n_features);
                for f in 0..n_features {
                    diff[f] = x[[i, f]] - mu_c[f];
                }
                let quad_form = diff.dot(&inv.dot(&diff));
                df[[i, c]] = -0.5 * log_det - 0.5 * quad_form + log_prior;
            }
        }

        df
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let df = self.decision_function(x);
        let classes = self.classes.as_ref().expect("Model not fitted");
        let (n_samples, n_classes) = df.dim();
        let mut preds = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_c = 0;
            for c in 0..n_classes {
                if df[[i, c]] > best_val {
                    best_val = df[[i, c]];
                    best_c = c;
                }
            }
            preds[i] = classes[best_c];
        }

        preds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lda_fit_predict_transform() {
        let x = array![
            [1.0, 1.0],
            [1.2, 0.9],
            [0.8, 1.1],
            [8.0, 8.0],
            [8.2, 7.9],
            [7.9, 8.1]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut lda = LinearDiscriminantAnalysis::new(Some(1));
        lda.fit(&x, &y).unwrap();

        let preds = lda.predict(&x);
        assert_eq!(preds, y);

        let x_trans = lda.transform(&x);
        assert_eq!(x_trans.dim(), (6, 1));
        assert!((x_trans[[0, 0]] - x_trans[[3, 0]]).abs() > 1.0);
    }

    #[test]
    fn test_qda_fit_predict() {
        let x = array![
            [1.0, 1.0],
            [1.2, 0.9],
            [0.8, 1.1],
            [8.0, 8.0],
            [8.2, 7.9],
            [7.9, 8.1]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();

        let preds = qda.predict(&x);
        assert_eq!(preds, y);
    }
}
