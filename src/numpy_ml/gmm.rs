use ndarray::{Array1, Array2, Array3, Axis};
use rand::{Rng, SeedableRng};

/// Gaussian Mixture Model trained via expectation-maximization.
pub struct GMM {
    pub c: usize,
    pub seed: Option<u64>,
    pub pi: Option<Array1<f64>>,
    pub q: Option<Array2<f64>>,
    pub mu: Option<Array2<f64>>,
    pub sigma: Option<Array3<f64>>,
    pub elbo: Option<f64>,
    pub is_fit: bool,
}

impl GMM {
    /// Create a new GMM with `c` mixture components and an optional random seed.
    pub fn new(c: usize, seed: Option<u64>) -> Self {
        Self {
            c,
            seed,
            pi: None,
            q: None,
            mu: None,
            sigma: None,
            elbo: None,
            is_fit: false,
        }
    }

    fn initialize_params(&mut self, x: &Array2<f64>) {
        let (n, d) = x.dim();
        let c = self.c;

        let mut rng = if let Some(s) = self.seed {
            rand::rngs::StdRng::seed_from_u64(s)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let rr: Vec<f64> = (0..c).map(|_| rng.r#gen::<f64>()).collect();
        let sum_rr: f64 = rr.iter().sum();
        let pi = Array1::from_vec(rr.into_iter().map(|v| v / sum_rr).collect());

        let mut mu = Array2::zeros((c, d));
        for i in 0..c {
            for j in 0..d {
                mu[[i, j]] = rng.gen_range(-5.0..10.0);
            }
        }

        let mut sigma = Array3::zeros((c, d, d));
        for i in 0..c {
            for j in 0..d {
                sigma[[i, j, j]] = 1.0;
            }
        }

        self.pi = Some(pi);
        self.q = Some(Array2::zeros((n, c)));
        self.mu = Some(mu);
        self.sigma = Some(sigma);
        self.elbo = None;
        self.is_fit = false;
    }

    /// Compute the variational lower bound (ELBO) under current parameters.
    pub fn likelihood_lower_bound(&self, x: &Array2<f64>) -> Result<f64, String> {
        let (n, _d) = x.dim();
        let c = self.c;
        let pi = self.pi.as_ref().ok_or("Model not fitted")?;
        let q = self.q.as_ref().ok_or("Model not fitted")?;
        let mu = self.mu.as_ref().ok_or("Model not fitted")?;
        let sigma = self.sigma.as_ref().ok_or("Model not fitted")?;

        let eps = f64::EPSILON;
        let mut expec1 = 0.0;
        let mut expec2 = 0.0;

        for i in 0..n {
            let x_i = x.row(i);
            for j in 0..c {
                let log_pi_k = (pi[j] + eps).ln();
                let log_p_x_i = log_gaussian_pdf(&x_i.to_owned(), &mu.row(j).to_owned(), &sigma.slice(ndarray::s![j, .., ..]).to_owned())?;
                let z_nk = q[[i, j]];

                expec1 += z_nk * (log_p_x_i + log_pi_k);
                expec2 += z_nk * (z_nk + eps).ln();
            }
        }

        Ok(expec1 - expec2)
    }

    /// Fit the GMM parameters using EM.
    pub fn fit(&mut self, x: &Array2<f64>, max_iter: usize, tol: f64, verbose: bool) -> Result<i32, String> {
        let mut prev_vlb = f64::NEG_INFINITY;
        self.initialize_params(x);

        for iter in 0..max_iter {
            self.e_step(x)?;
            self.m_step(x)?;
            let vlb = self.likelihood_lower_bound(x)?;

            if verbose {
                println!("{}. Lower bound: {}", iter + 1, vlb);
            }

            let converged = iter > 0 && (vlb - prev_vlb).abs() <= tol;
            if vlb.is_nan() || converged {
                break;
            }

            prev_vlb = vlb;
        }

        self.elbo = Some(prev_vlb);
        self.is_fit = true;
        Ok(0)
    }

    /// Predict log probabilities or hard cluster assignments for `x`.
    pub fn predict(&self, x: &Array2<f64>, soft_labels: bool) -> Result<ndarray::ArrayD<f64>, String> {
        if !self.is_fit {
            return Err("Must call .fit before making predictions".to_string());
        }

        let (m, _d) = x.dim();
        let c = self.c;
        let mu = self.mu.as_ref().ok_or("Model not fitted")?;
        let sigma = self.sigma.as_ref().ok_or("Model not fitted")?;

        if soft_labels {
            let mut result = Array2::zeros((m, c));
            for i in 0..m {
                let x_i = x.row(i).to_owned();
                for j in 0..c {
                    result[[i, j]] = log_gaussian_pdf(&x_i, &mu.row(j).to_owned(), &sigma.slice(ndarray::s![j, .., ..]).to_owned())?;
                }
            }
            Ok(result.into_dyn())
        } else {
            let mut result = Array1::zeros(m);
            for i in 0..m {
                let x_i = x.row(i).to_owned();
                let mut best_log_prob = f64::NEG_INFINITY;
                let mut best_idx = 0;
                for j in 0..c {
                    let log_prob = log_gaussian_pdf(&x_i, &mu.row(j).to_owned(), &sigma.slice(ndarray::s![j, .., ..]).to_owned())?;
                    if log_prob > best_log_prob {
                        best_log_prob = log_prob;
                        best_idx = j;
                    }
                }
                result[i] = best_idx as f64;
            }
            Ok(result.into_dyn())
        }
    }

    fn e_step(&mut self, x: &Array2<f64>) -> Result<(), String> {
        let (n, _d) = x.dim();
        let c = self.c;
        let pi = self.pi.as_ref().ok_or("Model not fitted")?;
        let mu = self.mu.as_ref().ok_or("Model not fitted")?;
        let sigma = self.sigma.as_ref().ok_or("Model not fitted")?;
        let q = self.q.as_mut().ok_or("Model not fitted")?;

        for i in 0..n {
            let x_i = x.row(i).to_owned();
            let mut denom_vals = Vec::with_capacity(c);
            for j in 0..c {
                let log_pi_c = pi[j].ln();
                let log_p_x_i = log_gaussian_pdf(&x_i, &mu.row(j).to_owned(), &sigma.slice(ndarray::s![j, .., ..]).to_owned())?;
                denom_vals.push(log_p_x_i + log_pi_c);
            }

            let log_denom = crate::numpy_ml::utils::logsumexp(&denom_vals);
            for j in 0..c {
                q[[i, j]] = (denom_vals[j] - log_denom).exp();
            }
        }

        Ok(())
    }

    fn m_step(&mut self, x: &Array2<f64>) -> Result<(), String> {
        let (n, d) = x.dim();
        let c = self.c;
        let q = self.q.as_ref().ok_or("Model not fitted")?;
        let mu = self.mu.as_mut().ok_or("Model not fitted")?;
        let sigma = self.sigma.as_mut().ok_or("Model not fitted")?;
        let pi = self.pi.as_mut().ok_or("Model not fitted")?;

        let denoms = q.sum_axis(Axis(0));

        // Update cluster priors
        for j in 0..c {
            pi[j] = denoms[j] / n as f64;
        }

        // Update cluster means
        for j in 0..c {
            let mut num = Array1::<f64>::zeros(d);
            for i in 0..n {
                num = num + q[[i, j]] * x.row(i).to_owned();
            }
            if denoms[j] > 0.0 {
                for k in 0..d {
                    mu[[j, k]] = num[k] / denoms[j];
                }
            } else {
                for k in 0..d {
                    mu[[j, k]] = 0.0;
                }
            }
        }

        // Update cluster covariances
        for j in 0..c {
            let mu_c = mu.row(j).to_owned();
            let n_c = denoms[j];
            let mut outer = Array2::zeros((d, d));

            for i in 0..n {
                let wic = q[[i, j]];
                let xi = x.row(i).to_owned();
                let diff = &xi - &mu_c;
                for a in 0..d {
                    for b in 0..d {
                        outer[[a, b]] += wic * diff[a] * diff[b];
                    }
                }
            }

            if n_c > 0.0 {
                outer.mapv_inplace(|v| v / n_c);
            }
            for a in 0..d {
                outer[[a, a]] += 1e-6;
            }

            for a in 0..d {
                for b in 0..d {
                    sigma[[j, a, b]] = outer[[a, b]];
                }
            }
        }

        Ok(())
    }
}

fn log_gaussian_pdf(x: &Array1<f64>, mu: &Array1<f64>, sigma: &Array2<f64>) -> Result<f64, String> {
    let n = mu.len();
    let a = n as f64 * (2.0 * std::f64::consts::PI).ln();
    let log_det = logdet(sigma)?;

    let diff = x - mu;
    let y = crate::sp::linalg::solve_vec(sigma, &diff)?;
    let c = diff.dot(&y);

    Ok(-0.5 * (a + log_det + c))
}

fn logdet(sigma: &Array2<f64>) -> Result<f64, String> {
    let svd = crate::sp::linalg::svd(sigma);
    let mut log_det = 0.0;
    for &s in svd.s.iter() {
        if s <= 0.0 {
            return Err("Covariance matrix is singular or indefinite".to_string());
        }
        log_det += s.ln();
    }
    Ok(log_det)
}
