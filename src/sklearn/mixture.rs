use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Gaussian Mixture Model (GMM).
///
/// Representation of a Gaussian mixture model probability distribution.
/// Estimation of model parameters is performed by Expectation-Maximization (EM).
pub struct GaussianMixture {
    pub n_components: usize,
    pub covariance_type: String,
    pub max_iter: usize,
    pub tol: f64,
    pub weights: Option<Array1<f64>>,
    pub means: Option<Array2<f64>>, // (n_components, n_features)
    pub covariances: Option<Vec<Array2<f64>>>,
    pub converged: bool,
    pub lower_bound: f64,
}

impl GaussianMixture {
    pub fn new(
        n_components: usize,
        covariance_type: Option<&str>,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> Self {
        Self {
            n_components,
            covariance_type: covariance_type.unwrap_or("full").to_string(),
            max_iter: max_iter.unwrap_or(100),
            tol: tol.unwrap_or(1e-3),
            weights: None,
            means: None,
            covariances: None,
            converged: false,
            lower_bound: f64::NEG_INFINITY,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
        let (n_samples, n_features) = x.dim();
        let k = self.n_components;
        if n_samples < k {
            return Err("n_samples must be >= n_components".to_string());
        }

        // Initialize weights uniformly
        let mut weights = Array1::from_elem(k, 1.0 / k as f64);

        // Initialize means from data quantiles or uniform spacing
        let mut means = Array2::zeros((k, n_features));
        let step = n_samples / k;
        for c in 0..k {
            let idx = (c * step).min(n_samples - 1);
            for f in 0..n_features {
                means[[c, f]] = x[[idx, f]];
            }
        }

        // Initialize covariances to identity (or sample variance)
        let sample_var = x.var_axis(ndarray::Axis(0), 1.0);
        let mut covariances = Vec::with_capacity(k);
        for _ in 0..k {
            let mut cov = Array2::zeros((n_features, n_features));
            for f in 0..n_features {
                cov[[f, f]] = sample_var[f].max(1e-4);
            }
            covariances.push(cov);
        }

        let mut prev_ll = f64::NEG_INFINITY;
        let mut converged = false;

        for _ in 0..self.max_iter {
            // --- E-step ---
            // Compute responsibilities gamma(i, c) = w_c * N(x_i | mu_c, Sigma_c) / sum_j ...
            let mut resp = Array2::zeros((n_samples, k));
            let mut sample_log_prob = Array1::zeros(n_samples);

            for c in 0..k {
                let cov = &covariances[c];
                let inv = crate::sp::linalg::inv(cov).unwrap_or_else(|_| Array2::eye(n_features));
                let det = crate::sp::linalg::det(cov).max(1e-12);
                let norm_const = 1.0 / ((2.0 * PI).powf(n_features as f64 / 2.0) * det.sqrt());
                let mu_c = means.row(c);

                for i in 0..n_samples {
                    let mut diff = Array1::zeros(n_features);
                    for f in 0..n_features {
                        diff[f] = x[[i, f]] - mu_c[f];
                    }
                    let mahal = diff.dot(&inv.dot(&diff));
                    let pdf = norm_const * (-0.5 * mahal).exp();
                    resp[[i, c]] = weights[c] * pdf;
                }
            }

            let mut cur_ll = 0.0;
            for i in 0..n_samples {
                let row_sum: f64 = resp.row(i).sum();
                sample_log_prob[i] = row_sum.max(1e-15).ln();
                cur_ll += sample_log_prob[i];
                if row_sum > 1e-15 {
                    for c in 0..k {
                        resp[[i, c]] /= row_sum;
                    }
                } else {
                    for c in 0..k {
                        resp[[i, c]] = 1.0 / k as f64;
                    }
                }
            }

            cur_ll /= n_samples as f64;

            if (cur_ll - prev_ll).abs() < self.tol {
                converged = true;
                prev_ll = cur_ll;
                break;
            }
            prev_ll = cur_ll;

            // --- M-step ---
            let mut n_k = Array1::zeros(k);
            for c in 0..k {
                n_k[c] = resp.column(c).sum();
                weights[c] = n_k[c] / n_samples as f64;

                // Update means
                for f in 0..n_features {
                    let mut num = 0.0;
                    for i in 0..n_samples {
                        num += resp[[i, c]] * x[[i, f]];
                    }
                    means[[c, f]] = num / n_k[c].max(1e-12);
                }

                // Update covariances
                let mut cov = Array2::zeros((n_features, n_features));
                let mu_c = means.row(c);
                for i in 0..n_samples {
                    let r = resp[[i, c]];
                    if self.covariance_type == "diag" {
                        for f in 0..n_features {
                            cov[[f, f]] += r * (x[[i, f]] - mu_c[f]).powi(2);
                        }
                    } else {
                        for f1 in 0..n_features {
                            let d1 = x[[i, f1]] - mu_c[f1];
                            for f2 in 0..n_features {
                                let d2 = x[[i, f2]] - mu_c[f2];
                                cov[[f1, f2]] += r * d1 * d2;
                            }
                        }
                    }
                }

                for f1 in 0..n_features {
                    for f2 in 0..n_features {
                        cov[[f1, f2]] /= n_k[c].max(1e-12);
                    }
                    cov[[f1, f1]] += 1e-6; // Regularization
                }
                covariances[c] = cov;
            }
        }

        self.weights = Some(weights);
        self.means = Some(means);
        self.covariances = Some(covariances);
        self.converged = converged;
        self.lower_bound = prev_ll;
        Ok(())
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let (n_samples, n_features) = x.dim();
        let k = self.n_components;
        let weights = self.weights.as_ref().expect("Model not fitted");
        let means = self.means.as_ref().expect("Model not fitted");
        let covariances = self.covariances.as_ref().expect("Model not fitted");

        let mut resp = Array2::zeros((n_samples, k));
        for c in 0..k {
            let cov = &covariances[c];
            let inv = crate::sp::linalg::inv(cov).unwrap_or_else(|_| Array2::eye(n_features));
            let det = crate::sp::linalg::det(cov).max(1e-12);
            let norm_const = 1.0 / ((2.0 * PI).powf(n_features as f64 / 2.0) * det.sqrt());
            let mu_c = means.row(c);

            for i in 0..n_samples {
                let mut diff = Array1::zeros(n_features);
                for f in 0..n_features {
                    diff[f] = x[[i, f]] - mu_c[f];
                }
                let mahal = diff.dot(&inv.dot(&diff));
                let pdf = norm_const * (-0.5 * mahal).exp();
                resp[[i, c]] = weights[c] * pdf;
            }
        }

        for i in 0..n_samples {
            let row_sum: f64 = resp.row(i).sum();
            if row_sum > 1e-15 {
                for c in 0..k {
                    resp[[i, c]] /= row_sum;
                }
            } else {
                for c in 0..k {
                    resp[[i, c]] = 1.0 / k as f64;
                }
            }
        }
        resp
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let proba = self.predict_proba(x);
        let (n_samples, k) = proba.dim();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_c = 0;
            for c in 0..k {
                if proba[[i, c]] > best_val {
                    best_val = proba[[i, c]];
                    best_c = c;
                }
            }
            labels[i] = best_c as f64;
        }

        labels
    }

    pub fn score_samples(&self, x: &Array2<f64>) -> Array1<f64> {
        let (n_samples, n_features) = x.dim();
        let k = self.n_components;
        let weights = self.weights.as_ref().expect("Model not fitted");
        let means = self.means.as_ref().expect("Model not fitted");
        let covariances = self.covariances.as_ref().expect("Model not fitted");

        let mut scores = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut sample_pdf = 0.0;
            for c in 0..k {
                let cov = &covariances[c];
                let inv = crate::sp::linalg::inv(cov).unwrap_or_else(|_| Array2::eye(n_features));
                let det = crate::sp::linalg::det(cov).max(1e-12);
                let norm_const = 1.0 / ((2.0 * PI).powf(n_features as f64 / 2.0) * det.sqrt());
                let mu_c = means.row(c);

                let mut diff = Array1::zeros(n_features);
                for f in 0..n_features {
                    diff[f] = x[[i, f]] - mu_c[f];
                }
                let mahal = diff.dot(&inv.dot(&diff));
                sample_pdf += weights[c] * norm_const * (-0.5 * mahal).exp();
            }
            scores[i] = sample_pdf.max(1e-20).ln();
        }
        scores
    }

    pub fn score(&self, x: &Array2<f64>) -> f64 {
        self.score_samples(x).mean().unwrap_or(0.0)
    }

    /// Bayesian Information Criterion (BIC): -2 * ln(L) + k_params * ln(N).
    pub fn bic(&self, x: &Array2<f64>) -> f64 {
        let (n_samples, n_features) = x.dim();
        let k = self.n_components;
        let n_params = (k - 1)
            + k * n_features
            + if self.covariance_type == "diag" {
                k * n_features
            } else {
                k * n_features * (n_features + 1) / 2
            };

        let total_log_l = self.score_samples(x).sum();
        -2.0 * total_log_l + (n_params as f64) * (n_samples as f64).ln()
    }

    /// Akaike Information Criterion (AIC): -2 * ln(L) + 2 * k_params.
    pub fn aic(&self, x: &Array2<f64>) -> f64 {
        let (_, n_features) = x.dim();
        let k = self.n_components;
        let n_params = (k - 1)
            + k * n_features
            + if self.covariance_type == "diag" {
                k * n_features
            } else {
                k * n_features * (n_features + 1) / 2
            };

        let total_log_l = self.score_samples(x).sum();
        -2.0 * total_log_l + 2.0 * (n_params as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gaussian_mixture_clustering() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0]
        ];

        let mut gmm = GaussianMixture::new(2, Some("full"), Some(100), Some(1e-4));
        gmm.fit(&x).unwrap();

        let labels = gmm.predict(&x);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);

        assert!(gmm.score(&x).is_finite());
        assert!(gmm.bic(&x).is_finite());
        assert!(gmm.aic(&x).is_finite());
    }
}
