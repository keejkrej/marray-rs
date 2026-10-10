use ndarray::{Array1, Array2, Axis};

/// Principal Component Analysis (PCA).
pub struct PCA {
    pub n_components: usize,
    pub components: Option<Array2<f64>>,
    pub mean: Option<Array1<f64>>,
    pub explained_variance: Option<Array1<f64>>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            mean: None,
            explained_variance: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
        let (nrows, ncols) = x.dim();
        if nrows <= 1 {
            return Err("PCA requires at least 2 samples".to_string());
        }

        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut x_centered = Array2::zeros((nrows, ncols));
        for r in 0..nrows {
            for c in 0..ncols {
                x_centered[[r, c]] = x[[r, c]] - mean[c];
            }
        }

        let svd_res = crate::sp::linalg::svd(&x_centered);
        let vh = svd_res.vh.ok_or_else(|| "SVD failed to compute V^T".to_string())?;

        let k = std::cmp::min(self.n_components, ncols);
        let mut components = Array2::zeros((k, ncols));
        for r in 0..k {
            for c in 0..ncols {
                components[[r, c]] = vh[[r, c]];
            }
        }

        let s = svd_res.s;
        let mut explained_variance = Array1::zeros(k);
        for i in 0..k {
            explained_variance[i] = s[i].powi(2) / ((nrows - 1) as f64);
        }

        self.mean = Some(mean);
        self.components = Some(components);
        self.explained_variance = Some(explained_variance);
        Ok(())
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().expect("PCA model not fitted");
        let components = self.components.as_ref().expect("PCA model not fitted");
        let (nrows, ncols) = x.dim();
        let mut x_centered = Array2::zeros((nrows, ncols));
        for r in 0..nrows {
            for c in 0..ncols {
                x_centered[[r, c]] = x[[r, c]] - mean[c];
            }
        }
        x_centered.dot(&components.t())
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        self.fit(x)?;
        Ok(self.transform(x))
    }
}

/// Non-Negative Matrix Factorization (NMF).
///
/// Factors non-negative matrix $X \approx W H$ using Lee & Seung multiplicative updates.
pub struct NMF {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub components: Option<Array2<f64>>, // H matrix: (n_components, n_features)
    pub reconstruction_err: Option<f64>,
}

impl NMF {
    pub fn new(n_components: usize, max_iter: Option<usize>, tol: Option<f64>) -> Self {
        Self {
            n_components,
            max_iter: max_iter.unwrap_or(200),
            tol: tol.unwrap_or(1e-4),
            components: None,
            reconstruction_err: None,
        }
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (n_samples, n_features) = x.dim();
        if x.iter().any(|&v| v < 0.0) {
            return Err("NMF requires non-negative input values".to_string());
        }

        let k = self.n_components;
        let mut rng = rand::thread_rng();
        use rand::Rng;

        // Initialize W and H with small positive values
        let mut w = Array2::from_shape_fn((n_samples, k), |_| rng.gen_range(0.1..1.0));
        let mut h = Array2::from_shape_fn((k, n_features), |_| rng.gen_range(0.1..1.0));

        let eps = 1e-12;
        let mut prev_err = f64::INFINITY;

        for _ in 0..self.max_iter {
            // Update H: H *= (W^T X) / (W^T W H + eps)
            let wt = w.t();
            let wtx = wt.dot(x);
            let wtwh = wt.dot(&w).dot(&h) + eps;
            for i in 0..k {
                for j in 0..n_features {
                    h[[i, j]] *= wtx[[i, j]] / wtwh[[i, j]];
                }
            }

            // Update W: W *= (X H^T) / (W H H^T + eps)
            let ht = h.t();
            let xht = x.dot(&ht);
            let whht = w.dot(&h).dot(&ht) + eps;
            for i in 0..n_samples {
                for j in 0..k {
                    w[[i, j]] *= xht[[i, j]] / whht[[i, j]];
                }
            }

            // Compute Frobenius reconstruction error: ||X - W H||_F
            let diff = x - &w.dot(&h);
            let err = diff.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if (prev_err - err).abs() < self.tol {
                prev_err = err;
                break;
            }
            prev_err = err;
        }

        self.reconstruction_err = Some(prev_err);
        self.components = Some(h);
        Ok(w)
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
        self.fit_transform(x)?;
        Ok(())
    }

    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let h = self.components.as_ref().ok_or("Model not fitted")?;
        let (n_samples, _) = x.dim();
        let k = self.n_components;
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let mut w = Array2::from_shape_fn((n_samples, k), |_| rng.gen_range(0.1..1.0));
        let eps = 1e-12;
        let ht = h.t();
        let xht = x.dot(&ht);

        for _ in 0..self.max_iter {
            let whht = w.dot(h).dot(&ht) + eps;
            for i in 0..n_samples {
                for j in 0..k {
                    w[[i, j]] *= xht[[i, j]] / whht[[i, j]];
                }
            }
        }
        Ok(w)
    }
}

/// Dimensionality reduction using truncated SVD (also known as LSA).
pub struct TruncatedSVD {
    pub n_components: usize,
    pub components: Option<Array2<f64>>,
    pub singular_values: Option<Array1<f64>>,
    pub explained_variance: Option<Array1<f64>>,
}

impl TruncatedSVD {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            singular_values: None,
            explained_variance: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
        let (nrows, ncols) = x.dim();
        let k = self.n_components.min(ncols).min(nrows);

        let svd_res = crate::sp::linalg::svd(x);
        let vh = svd_res.vh.ok_or_else(|| "SVD failed".to_string())?;

        let mut components = Array2::zeros((k, ncols));
        for r in 0..k {
            for c in 0..ncols {
                components[[r, c]] = vh[[r, c]];
            }
        }

        let mut s = Array1::zeros(k);
        let mut exp_var = Array1::zeros(k);
        for i in 0..k {
            s[i] = svd_res.s[i];
            exp_var[i] = s[i].powi(2) / ((nrows - 1).max(1) as f64);
        }

        self.components = Some(components);
        self.singular_values = Some(s);
        self.explained_variance = Some(exp_var);
        Ok(())
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let components = self.components.as_ref().expect("Model not fitted");
        x.dot(&components.t())
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        self.fit(x)?;
        Ok(self.transform(x))
    }
}

/// Factor Analysis (FA) using Expectation-Maximization.
pub struct FactorAnalysis {
    pub n_components: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub components: Option<Array2<f64>>, // Loadings L: (n_components, n_features)
    pub mean: Option<Array1<f64>>,
    pub noise_variance: Option<Array1<f64>>,
}

impl FactorAnalysis {
    pub fn new(n_components: usize, max_iter: Option<usize>, tol: Option<f64>) -> Self {
        Self {
            n_components,
            max_iter: max_iter.unwrap_or(100),
            tol: tol.unwrap_or(1e-4),
            components: None,
            mean: None,
            noise_variance: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
        let (n_samples, n_features) = x.dim();
        if n_samples <= 1 {
            return Err("FactorAnalysis requires at least 2 samples".to_string());
        }

        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut xc = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                xc[[i, j]] = x[[i, j]] - mean[j];
            }
        }

        let k = self.n_components;
        let mut rng = rand::thread_rng();
        use rand::Rng;

        // Initialize loadings L (k, p) and noise variances psi (p)
        let mut l = Array2::from_shape_fn((k, n_features), |_| rng.gen_range(-0.1..0.1));
        let sample_var = xc.var_axis(Axis(0), 1.0);
        let mut psi = sample_var.clone();
        for v in psi.iter_mut() {
            *v = v.max(1e-4);
        }

        // EM loop
        let sample_cov = xc.t().dot(&xc) / (n_samples as f64);

        for _ in 0..self.max_iter {
            // M = I + L psi^{-1} L^T
            let mut lt_psi_inv = Array2::zeros((n_features, k));
            for j in 0..n_features {
                let inv_p = 1.0 / psi[j];
                for i in 0..k {
                    lt_psi_inv[[j, i]] = l[[i, j]] * inv_p;
                }
            }

            let mut m = Array2::eye(k);
            let l_lt_psi = l.dot(&lt_psi_inv);
            m = &m + &l_lt_psi;

            let m_inv = match crate::sp::linalg::inv(&m) {
                Ok(inv) => inv,
                Err(_) => break,
            };

            // beta = L^T (L L^T + Psi)^{-1} = M^{-1} L Psi^{-1}
            let beta = m_inv.dot(&l.dot(&Array2::from_diag(&psi.mapv(|v| 1.0 / v))));

            // New loadings L_new = Cov * beta^T * (I - beta * L + beta * Cov * beta^T)^{-1}
            let term = Array2::eye(k) - beta.dot(&l.t()) + beta.dot(&sample_cov).dot(&beta.t());
            let term_inv = match crate::sp::linalg::inv(&term) {
                Ok(inv) => inv,
                Err(_) => break,
            };

            let l_new = sample_cov.dot(&beta.t()).dot(&term_inv).t().to_owned();

            // New psi
            let l_new_beta = l_new.t().dot(&beta);
            let mut psi_new = Array1::zeros(n_features);
            for j in 0..n_features {
                let diag_val = sample_cov[[j, j]] - l_new_beta.row(j).dot(&sample_cov.row(j));
                psi_new[j] = diag_val.max(1e-4);
            }

            let diff = (&l_new - &l).iter().map(|&v| v * v).sum::<f64>().sqrt();
            l = l_new;
            psi = psi_new;

            if diff < self.tol {
                break;
            }
        }

        self.mean = Some(mean);
        self.components = Some(l);
        self.noise_variance = Some(psi);
        Ok(())
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().expect("Model not fitted");
        let l = self.components.as_ref().expect("Model not fitted");
        let psi = self.noise_variance.as_ref().expect("Model not fitted");
        let (n_samples, n_features) = x.dim();

        let mut xc = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                xc[[i, j]] = x[[i, j]] - mean[j];
            }
        }

        // Expected factors: E[f|x] = (I + L Psi^{-1} L^T)^{-1} L Psi^{-1} (x - mu)
        let k = self.n_components;
        let mut lt_psi_inv = Array2::zeros((n_features, k));
        for j in 0..n_features {
            let inv_p = 1.0 / psi[j];
            for i in 0..k {
                lt_psi_inv[[j, i]] = l[[i, j]] * inv_p;
            }
        }

        let m = Array2::eye(k) + l.dot(&lt_psi_inv);
        let m_inv = crate::sp::linalg::inv(&m).unwrap_or_else(|_| Array2::eye(k));

        let l_psi_inv = l.dot(&Array2::from_diag(&psi.mapv(|v| 1.0 / v)));
        let weight = m_inv.dot(&l_psi_inv);

        xc.dot(&weight.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_nmf() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        let mut nmf = NMF::new(2, Some(100), None);
        let w = nmf.fit_transform(&x).unwrap();
        assert_eq!(w.dim(), (3, 2));
        assert!(nmf.components.is_some());
        assert!(nmf.reconstruction_err.unwrap() < 1.0);
    }

    #[test]
    fn test_truncated_svd() {
        let x = array![
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ];
        let mut svd = TruncatedSVD::new(2);
        let xt = svd.fit_transform(&x).unwrap();
        assert_eq!(xt.dim(), (3, 2));
        assert!(svd.singular_values.is_some());
    }

    #[test]
    fn test_factor_analysis() {
        let x = array![
            [1.0, 2.0, 1.1],
            [2.0, 4.0, 2.0],
            [3.0, 6.0, 3.2],
            [4.0, 8.0, 3.9]
        ];
        let mut fa = FactorAnalysis::new(1, Some(50), None);
        fa.fit(&x).unwrap();
        let factors = fa.transform(&x);
        assert_eq!(factors.dim(), (4, 1));
    }
}

