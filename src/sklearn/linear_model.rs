use ndarray::{Array1, Array2};

/// Ordinary least squares Linear Regression.
pub struct LinearRegression {
    pub coef: Option<Array1<f64>>,
    pub intercept: Option<f64>,
    pub fit_intercept: bool,
}

impl LinearRegression {
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            coef: None,
            intercept: None,
            fit_intercept,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (nrows, ncols) = x.dim();
        if nrows == 0 || nrows != y.len() {
            return Err("Dimensions of X and y must match".to_string());
        }

        if self.fit_intercept {
            let mut x_design = Array2::ones((nrows, ncols + 1));
            for r in 0..nrows {
                for c in 0..ncols {
                    x_design[[r, c]] = x[[r, c]];
                }
            }
            let xt = x_design.t().to_owned();
            let xtx = xt.dot(&x_design);
            let xty = xt.dot(y);
            let beta = crate::sp::linalg::solve_vec(&xtx, &xty)?;

            self.coef = Some(Array1::from_vec(beta.slice(ndarray::s![0..ncols]).to_vec()));
            self.intercept = Some(beta[ncols]);
        } else {
            let xt = x.t().to_owned();
            let xtx = xt.dot(x);
            let xty = xt.dot(y);
            let beta = crate::sp::linalg::solve_vec(&xtx, &xty)?;

            self.coef = Some(beta);
            self.intercept = Some(0.0);
        }
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let coef = self.coef.as_ref().expect("LinearRegression model not fitted");
        let intercept = self.intercept.expect("LinearRegression model not fitted");
        let mut preds = x.dot(coef);
        preds.mapv_inplace(|v| v + intercept);
        preds
    }
}

/// Ridge Regression (L2-regularized Ordinary Least Squares).
pub struct Ridge {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub coef: Option<Array1<f64>>,
    pub intercept: Option<f64>,
}

impl Ridge {
    pub fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            coef: None,
            intercept: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (nrows, ncols) = x.dim();
        if nrows == 0 || nrows != y.len() {
            return Err("Dimensions of X and y must match".to_string());
        }

        if self.fit_intercept {
            let x_mean = x.mean_axis(ndarray::Axis(0)).unwrap();
            let y_mean = y.mean().unwrap_or(0.0);

            let mut x_centered = Array2::zeros((nrows, ncols));
            for r in 0..nrows {
                for c in 0..ncols {
                    x_centered[[r, c]] = x[[r, c]] - x_mean[c];
                }
            }
            let y_centered = y - y_mean;

            let xt = x_centered.t().to_owned();
            let xtx = xt.dot(&x_centered);
            let xty = xt.dot(&y_centered);

            let mut reg_matrix: Array2<f64> = Array2::eye(ncols);
            reg_matrix.mapv_inplace(|v| v * self.alpha);
            let xtx_reg = xtx + reg_matrix;

            let w = crate::sp::linalg::solve_vec(&xtx_reg, &xty)?;

            let mut dot_product = 0.0;
            for i in 0..ncols {
                dot_product += w[i] * x_mean[i];
            }

            self.coef = Some(w);
            self.intercept = Some(y_mean - dot_product);
        } else {
            let xt = x.t().to_owned();
            let xtx = xt.dot(x);
            let xty = xt.dot(y);

            let mut reg_matrix: Array2<f64> = Array2::eye(ncols);
            reg_matrix.mapv_inplace(|v| v * self.alpha);
            let xtx_reg = xtx + reg_matrix;

            let w = crate::sp::linalg::solve_vec(&xtx_reg, &xty)?;

            self.coef = Some(w);
            self.intercept = Some(0.0);
        }
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let coef = self.coef.as_ref().expect("Ridge model not fitted");
        let intercept = self.intercept.expect("Ridge model not fitted");
        let mut preds = x.dot(coef);
        preds.mapv_inplace(|v| v + intercept);
        preds
    }
}

/// Bayesian linear regression with an unknown Gaussian error variance.
#[deprecated(note = "Use crate::numpy_ml::linear_models::BayesianLinearRegressionUnknownVariance instead")]
pub struct BayesianLinearRegressionUnknownVariance {
    pub alpha: f64,
    pub beta: f64,
    pub mu: f64,
    pub prior_cov: Option<Array2<f64>>,
    pub fit_intercept: bool,
    pub posterior_mean: Option<Array1<f64>>,
    pub posterior_cov: Option<Array2<f64>>,
}

#[allow(deprecated)]
impl BayesianLinearRegressionUnknownVariance {
    pub fn new(alpha: f64, beta: f64, mu: f64, prior_cov: Option<Array2<f64>>, fit_intercept: bool) -> Self {
        Self {
            alpha,
            beta,
            mu,
            prior_cov,
            fit_intercept,
            posterior_mean: None,
            posterior_cov: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (nrows, ncols) = x.dim();
        if nrows == 0 || nrows != y.len() {
            return Err("Dimensions of X and y must match".to_string());
        }

        let mut design = x.to_owned();
        let prior_mean = Array1::from_elem(if self.fit_intercept { ncols + 1 } else { ncols }, self.mu);
        if self.fit_intercept {
            let mut with_intercept = Array2::ones((nrows, ncols + 1));
            for r in 0..nrows {
                for c in 0..ncols {
                    with_intercept[[r, c + 1]] = x[[r, c]];
                }
            }
            design = with_intercept;
        }

        let m = design.ncols();
        let prior_cov = self.prior_cov.clone().unwrap_or_else(|| Array2::eye(m));
        if prior_cov.nrows() != m || prior_cov.ncols() != m {
            return Err("Prior covariance must match the design dimension".to_string());
        }

        let v_inv = crate::sp::linalg::inv(&prior_cov)?;
        let xt = design.t().to_owned();
        let lhs = v_inv.clone() + xt.dot(&design);
        let rhs = v_inv.dot(&prior_mean) + xt.dot(y);
        let posterior_mean = crate::sp::linalg::solve_vec(&lhs, &rhs)?;

        let x_mu = design.dot(&prior_mean);
        let a = y - &x_mu;
        let cov_term = design.dot(&prior_cov).dot(&design.t().to_owned()) + Array2::<f64>::eye(nrows);
        let b = crate::sp::linalg::inv(&cov_term)?;
        let c = y - &x_mu;
        let shape = nrows as f64 + self.alpha;
        let sigma = (1.0 / shape) * (self.alpha * self.beta.powi(2) + a.dot(&b.dot(&c)));
        let scale = sigma * sigma;
        let sigma_mode = scale / (shape - 1.0);
        let posterior_cov = crate::sp::linalg::inv(&lhs)? * sigma_mode;

        self.posterior_mean = Some(posterior_mean);
        self.posterior_cov = Some(posterior_cov);
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let posterior_mean = self.posterior_mean.as_ref().expect("Bayesian model not fitted");
        let mut design = x.to_owned();
        if self.fit_intercept {
            let (nrows, ncols) = x.dim();
            let mut with_intercept = Array2::ones((nrows, ncols + 1));
            for r in 0..nrows {
                for c in 0..ncols {
                    with_intercept[[r, c + 1]] = x[[r, c]];
                }
            }
            design = with_intercept;
        }
        design.dot(posterior_mean)
    }
}

/// Bayesian linear regression with a known Gaussian prior variance.
#[deprecated(note = "Use crate::numpy_ml::linear_models::BayesianLinearRegressionKnownVariance instead")]
pub struct BayesianLinearRegressionKnownVariance {
    pub mu: f64,
    pub sigma: f64,
    pub prior_cov: Option<Array2<f64>>,
    pub fit_intercept: bool,
    pub posterior_mean: Option<Array1<f64>>,
    pub posterior_cov: Option<Array2<f64>>,
}

#[allow(deprecated)]
impl BayesianLinearRegressionKnownVariance {
    pub fn new(mu: f64, sigma: f64, prior_cov: Option<Array2<f64>>, fit_intercept: bool) -> Self {
        Self {
            mu,
            sigma,
            prior_cov,
            fit_intercept,
            posterior_mean: None,
            posterior_cov: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (nrows, ncols) = x.dim();
        if nrows == 0 || nrows != y.len() {
            return Err("Dimensions of X and y must match".to_string());
        }

        let design = if self.fit_intercept {
            let mut design = Array2::ones((nrows, ncols + 1));
            for r in 0..nrows {
                for c in 0..ncols {
                    design[[r, c + 1]] = x[[r, c]];
                }
            }
            design
        } else {
            x.to_owned()
        };

        let m = design.ncols();
        let prior_cov = self.prior_cov.clone().unwrap_or_else(|| Array2::eye(m));
        if prior_cov.nrows() != m || prior_cov.ncols() != m {
            return Err("Prior covariance must match the design dimension".to_string());
        }

        let v_inv = crate::sp::linalg::inv(&prior_cov)?;
        let v_inv_mu = v_inv.dot(&Array1::from_elem(m, self.mu));
        let xt = design.t().to_owned();
        let lhs = v_inv + xt.dot(&design);
        let rhs = v_inv_mu + xt.dot(y);

        let posterior_mean = crate::sp::linalg::solve_vec(&lhs, &rhs)?;
        let sigma2 = self.sigma * self.sigma;
        let posterior_cov = crate::sp::linalg::inv(&lhs)? * sigma2;

        self.posterior_mean = Some(posterior_mean);
        self.posterior_cov = Some(posterior_cov);
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let posterior_mean = self.posterior_mean.as_ref().expect("Bayesian model not fitted");
        let mut design = x.to_owned();
        if self.fit_intercept {
            let (nrows, ncols) = x.dim();
            let mut with_intercept = Array2::ones((nrows, ncols + 1));
            for r in 0..nrows {
                for c in 0..ncols {
                    with_intercept[[r, c + 1]] = x[[r, c]];
                }
            }
            design = with_intercept;
        }
        design.dot(posterior_mean)
    }
}

/// Generalized linear model with IRLS fitting.
#[deprecated(note = "Use crate::numpy_ml::linear_models::GeneralizedLinearModel instead")]
pub struct GeneralizedLinearModel {
    pub link: String,
    pub fit_intercept: bool,
    pub tol: f64,
    pub max_iter: usize,
    pub beta: Option<Array1<f64>>,
    pub is_fit: bool,
}

#[allow(deprecated)]
impl GeneralizedLinearModel {
    pub fn new(link: &str, fit_intercept: bool, tol: f64, max_iter: usize) -> Self {
        Self {
            link: link.to_string(),
            fit_intercept,
            tol,
            max_iter,
            beta: None,
            is_fit: false,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (nrows, ncols) = x.dim();
        if nrows == 0 || nrows != y.len() {
            return Err("Dimensions of X and y must match".to_string());
        }

        let mut mu = Array1::from_elem(nrows, y.iter().copied().sum::<f64>() / nrows as f64);
        let mut eta = link_values(&self.link, &mu);
        let mut theta = theta_values(&self.link, &mu);
        let mut beta = Array1::zeros(if self.fit_intercept { ncols + 1 } else { ncols });

        for _ in 0..self.max_iter {
            let link_prime = link_prime_values(&self.link, &mu);
            let b_prime2 = b_prime2_values(&self.link, &theta);
            let z = eta.clone() + &(y.to_owned() - &mu) * &link_prime;
            let weights = Array1::from_shape_fn(nrows, |i| {
                1.0 / (b_prime2[i] * link_prime[i].powi(2) + 1e-12)
            });

            let design = if self.fit_intercept {
                let mut design = Array2::ones((nrows, ncols + 1));
                for r in 0..nrows {
                    for c in 0..ncols {
                        design[[r, c + 1]] = x[[r, c]];
                    }
                }
                design
            } else {
                x.to_owned()
            };

            let sqrt_weights = weights.mapv(|v| v.sqrt());
            let mut weighted_x = Array2::zeros((nrows, design.ncols()));
            for r in 0..nrows {
                for c in 0..design.ncols() {
                    weighted_x[[r, c]] = design[[r, c]] * sqrt_weights[r];
                }
            }
            let weighted_z = z * &sqrt_weights;

            let xtwx = weighted_x.t().dot(&weighted_x);
            let xtwz = weighted_x.t().dot(&weighted_z);
            let beta_new = crate::sp::linalg::solve_vec(&xtwx, &xtwz)?;

            let diff_now = (&beta_new - &beta).iter().map(|v| v.abs()).sum::<f64>();
            beta = beta_new;
            let diff = diff_now;
            eta = design.dot(&beta);
            mu = inv_link_values(&self.link, &eta);
            theta = theta_values(&self.link, &mu);

            if diff < self.tol * (beta.len() as f64) {
                break;
            }
        }

        self.beta = Some(beta);
        self.is_fit = true;
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let beta = self.beta.as_ref().expect("GeneralizedLinearModel not fitted");
        let (nrows, ncols) = x.dim();
        let design = if self.fit_intercept {
            let mut design = Array2::ones((nrows, ncols + 1));
            for r in 0..nrows {
                for c in 0..ncols {
                    design[[r, c + 1]] = x[[r, c]];
                }
            }
            design
        } else {
            x.to_owned()
        };
        inv_link_values(&self.link, &design.dot(beta))
    }
}

fn link_values(link: &str, mu: &Array1<f64>) -> Array1<f64> {
    let eps = f64::EPSILON;
    match link {
        "identity" => mu.clone(),
        "logit" => mu.iter().map(|m| (m + eps).ln() - (1.0 - m + eps).ln()).collect(),
        "log" => mu.iter().map(|m| (m + eps).ln()).collect(),
        _ => panic!("Unsupported link function: {link}"),
    }
}

fn inv_link_values(link: &str, eta: &Array1<f64>) -> Array1<f64> {
    match link {
        "identity" => eta.clone(),
        "logit" => eta.iter().map(|z| 1.0 / (1.0 + (-*z).exp())).collect(),
        "log" => eta.iter().map(|z| z.exp()).collect(),
        _ => panic!("Unsupported link function: {link}"),
    }
}

fn link_prime_values(link: &str, mu: &Array1<f64>) -> Array1<f64> {
    let eps = f64::EPSILON;
    match link {
        "identity" => Array1::from_elem(mu.len(), 1.0),
        "logit" => mu.iter().map(|m| 1.0 / (m + eps) + 1.0 / (1.0 - m + eps)).collect(),
        "log" => mu.iter().map(|m| 1.0 / (m + eps)).collect(),
        _ => panic!("Unsupported link function: {link}"),
    }
}

fn theta_values(link: &str, mu: &Array1<f64>) -> Array1<f64> {
    let eps = f64::EPSILON;
    match link {
        "identity" => mu.clone(),
        "logit" => mu.iter().map(|m| (m + eps).ln() - (1.0 - m + eps).ln()).collect(),
        "log" => mu.iter().map(|m| (m + eps).ln()).collect(),
        _ => panic!("Unsupported link function: {link}"),
    }
}

fn b_prime2_values(link: &str, theta: &Array1<f64>) -> Array1<f64> {
    match link {
        "identity" => Array1::from_elem(theta.len(), 1.0),
        "logit" => theta.iter().map(|t| (t.exp()) / ((1.0 + t.exp()).powi(2))).collect(),
        "log" => theta.iter().map(|t| t.exp()).collect(),
        _ => panic!("Unsupported link function: {link}"),
    }
}

pub struct LogisticRegression {
    pub coef: Option<Array1<f64>>,
    pub intercept: Option<f64>,
    pub max_iter: usize,
    pub lr: f64,
    pub tol: f64,
    pub c_reg: f64,
}

impl LogisticRegression {
    pub fn new(max_iter: usize, lr: f64, c_reg: f64) -> Self {
        Self {
            coef: None,
            intercept: None,
            max_iter,
            lr,
            tol: 1e-4,
            c_reg,
        }
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (nrows, ncols) = x.dim();
        let mut w = Array1::zeros(ncols);
        let mut b = 0.0;
        let lambda = 1.0 / self.c_reg;

        for _ in 0..self.max_iter {
            let mut z = x.dot(&w);
            z.mapv_inplace(|v| v + b);
            let p = z.mapv(Self::sigmoid);
            let diff = &p - y;

            let xt = x.t().to_owned();
            let mut dw = xt.dot(&diff) / (nrows as f64);
            if lambda > 0.0 {
                dw = dw + &w * lambda;
            }
            let db = diff.sum() / (nrows as f64);

            let mut max_grad_change = 0.0;
            for i in 0..ncols {
                let step = self.lr * dw[i];
                if step.abs() > max_grad_change {
                    max_grad_change = step.abs();
                }
                w[i] -= step;
            }
            let b_step = self.lr * db;
            if b_step.abs() > max_grad_change {
                max_grad_change = b_step.abs();
            }
            b -= b_step;

            if max_grad_change < self.tol {
                break;
            }
        }

        self.coef = Some(w);
        self.intercept = Some(b);
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let w = self.coef.as_ref().expect("LogisticRegression not fitted");
        let b = self.intercept.expect("LogisticRegression not fitted");
        let (nrows, _) = x.dim();
        let mut z = x.dot(w);
        z.mapv_inplace(|v| v + b);
        let p = z.mapv(Self::sigmoid);

        let mut proba = Array2::zeros((nrows, 2));
        for r in 0..nrows {
            proba[[r, 0]] = 1.0 - p[r];
            proba[[r, 1]] = p[r];
        }
        proba
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let proba = self.predict_proba(x);
        let (nrows, _) = proba.dim();
        let mut preds = Array1::zeros(nrows);
        for r in 0..nrows {
            preds[r] = if proba[[r, 1]] >= 0.5 { 1.0 } else { 0.0 };
        }
        preds
    }
}
