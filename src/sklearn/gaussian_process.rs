use ndarray::{Array1, Array2};

/// Gaussian Process Regressor (GPR).
pub struct GaussianProcessRegressor {
    pub alpha: f64,
    pub length_scale: f64,
    pub x_train: Option<Array2<f64>>,
    pub y_train: Option<Array1<f64>>,
    pub k_inv: Option<Array2<f64>>,
    pub alpha_weights: Option<Array1<f64>>,
}

impl GaussianProcessRegressor {
    pub fn new(alpha: Option<f64>, length_scale: Option<f64>) -> Self {
        Self {
            alpha: alpha.unwrap_or(1e-10),
            length_scale: length_scale.unwrap_or(1.0),
            x_train: None,
            y_train: None,
            k_inv: None,
            alpha_weights: None,
        }
    }

    fn rbf_kernel(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let (n1, d1) = x1.dim();
        let (n2, d2) = x2.dim();
        assert_eq!(d1, d2);

        let mut k = Array2::zeros((n1, n2));
        let l2 = 2.0 * self.length_scale.powi(2);

        for i in 0..n1 {
            for j in 0..n2 {
                let dist_sq: f64 = (0..d1).map(|d| (x1[[i, d]] - x2[[j, d]]).powi(2)).sum();
                k[[i, j]] = (-dist_sq / l2).exp();
            }
        }
        k
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let n = x.nrows();
        assert_eq!(n, y.len());

        let mut k = self.rbf_kernel(x, x);
        for i in 0..n {
            k[[i, i]] += self.alpha;
        }

        let k_inv = crate::sp::linalg::inv(&k)
            .map_err(|_| "Failed to invert kernel matrix (singular)".to_string())?;
        let alpha_weights = k_inv.dot(y);

        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        self.k_inv = Some(k_inv);
        self.alpha_weights = Some(alpha_weights);
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>, return_std: bool) -> (Array1<f64>, Option<Array1<f64>>) {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let alpha = self.alpha_weights.as_ref().expect("Model not fitted");
        let k_inv = self.k_inv.as_ref().expect("Model not fitted");

        let k_trans = self.rbf_kernel(x, x_train);
        let y_mean = k_trans.dot(alpha);

        let y_std = if return_std {
            let n_test = x.nrows();
            let mut std_vec = Array1::zeros(n_test);
            let k_self = self.rbf_kernel(x, x);
            for i in 0..n_test {
                let row = k_trans.row(i);
                let v = k_inv.dot(&row);
                let var = k_self[[i, i]] - row.dot(&v);
                std_vec[i] = var.max(0.0).sqrt();
            }
            Some(std_vec)
        } else {
            None
        };

        (y_mean, y_std)
    }
}

/// Gaussian Process Classifier (GPC) for binary classification using Laplace approximation.
pub struct GaussianProcessClassifier {
    pub length_scale: f64,
    pub max_iter: usize,
    pub x_train: Option<Array2<f64>>,
    pub y_train: Option<Array1<f64>>,
    pub f_latent: Option<Array1<f64>>,
    pub k_matrix: Option<Array2<f64>>,
}

impl GaussianProcessClassifier {
    pub fn new(length_scale: Option<f64>, max_iter: Option<usize>) -> Self {
        Self {
            length_scale: length_scale.unwrap_or(1.0),
            max_iter: max_iter.unwrap_or(100),
            x_train: None,
            y_train: None,
            f_latent: None,
            k_matrix: None,
        }
    }

    fn rbf_kernel(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let (n1, d1) = x1.dim();
        let (n2, _d2) = x2.dim();
        let mut k = Array2::zeros((n1, n2));
        let l2 = 2.0 * self.length_scale.powi(2);

        for i in 0..n1 {
            for j in 0..n2 {
                let dist_sq: f64 = (0..d1).map(|d| (x1[[i, d]] - x2[[j, d]]).powi(2)).sum();
                k[[i, j]] = (-dist_sq / l2).exp();
            }
        }
        k
    }

    #[inline]
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
    }

    /// Fit using Newton-Raphson to find the mode of the latent function f (Laplace approximation).
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let n = x.nrows();
        assert_eq!(n, y.len());

        let mut k = self.rbf_kernel(x, x);
        for i in 0..n {
            k[[i, i]] += 1e-5;
        }

        let mut f = Array1::<f64>::zeros(n);
        let k_inv = crate::sp::linalg::inv(&k).map_err(|_| "Failed to invert K".to_string())?;

        for _ in 0..self.max_iter {
            let pi = f.mapv(Self::sigmoid);
            let grad = y - &pi - &k_inv.dot(&f);

            // W = diag(pi * (1 - pi))
            let mut w = Array2::zeros((n, n));
            for i in 0..n {
                w[[i, i]] = (pi[i] * (1.0 - pi[i])).max(1e-6);
            }

            // Hessian H = -(K_inv + W)
            let h = &k_inv + &w;
            let h_inv = crate::sp::linalg::inv(&h).unwrap_or_else(|_| Array2::eye(n));
            let delta = h_inv.dot(&grad);

            f = &f + &delta;
            if delta.iter().map(|&d| d * d).sum::<f64>().sqrt() < 1e-4 {
                break;
            }
        }

        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        self.f_latent = Some(f);
        self.k_matrix = Some(k);
        Ok(())
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let k = self.k_matrix.as_ref().expect("Model not fitted");
        let f = self.f_latent.as_ref().expect("Model not fitted");

        let k_inv = crate::sp::linalg::inv(k).unwrap_or_else(|_| Array2::eye(k.nrows()));
        let k_star = self.rbf_kernel(x, x_train);

        let f_star = k_star.dot(&k_inv.dot(f));
        let n_test = x.nrows();
        let mut probas = Array2::zeros((n_test, 2));

        for i in 0..n_test {
            let p1 = Self::sigmoid(f_star[i]);
            probas[[i, 0]] = 1.0 - p1;
            probas[[i, 1]] = p1;
        }

        probas
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let probas = self.predict_proba(x);
        let n = x.nrows();
        let mut preds = Array1::zeros(n);
        for i in 0..n {
            preds[i] = if probas[[i, 1]] >= 0.5 { 1.0 } else { 0.0 };
        }
        preds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gpr() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let mut gpr = GaussianProcessRegressor::new(Some(1e-4), Some(1.0));
        gpr.fit(&x, &y).unwrap();

        let x_test = array![[1.0], [2.0]];
        let (mean, std) = gpr.predict(&x_test, true);
        assert!((mean[0] - 1.0).abs() < 0.1);
        assert!((mean[1] - 4.0).abs() < 0.1);
        assert!(std.is_some());
    }

    #[test]
    fn test_gpc() {
        let x = array![[0.0], [0.5], [2.0], [2.5]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut gpc = GaussianProcessClassifier::new(Some(1.0), Some(50));
        gpc.fit(&x, &y).unwrap();

        let preds = gpc.predict(&x);
        assert_eq!(preds, y);
    }
}
