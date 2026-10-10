use ndarray::{Array1, Array2, Array3, Array4};
use rand::Rng;
use rand::distributions::{Distribution, WeightedIndex};

/// A hidden Markov model with multinomial emission distribution.
pub struct MultinomialHMM {
    pub a: Option<Array2<f64>>,
    pub b: Option<Array2<f64>>,
    pub pi: Option<Array1<f64>>,
    pub eps: f64,
    pub n: Option<usize>,
    pub v: Option<usize>,
}

impl MultinomialHMM {
    /// Create a new HMM with optional transition matrix `a`, emission matrix `b`,
    /// and initial state distribution `pi`.
    pub fn new(a: Option<Array2<f64>>, b: Option<Array2<f64>>, pi: Option<Array1<f64>>, eps: Option<f64>) -> Self {
        let eps = eps.unwrap_or(f64::EPSILON);

        let n = a.as_ref().map(|a| a.nrows());
        let v = b.as_ref().map(|b| b.ncols());

        let a = a.map(|mut a| {
            a.mapv_inplace(|v| if v == 0.0 { eps } else { v });
            a
        });
        let b = b.map(|mut b| {
            b.mapv_inplace(|v| if v == 0.0 { eps } else { v });
            b
        });

        Self { a, b, pi, eps, n, v }
    }

    /// Sample a sequence of latent states and emissions of length `n_steps`.
    pub fn generate(&self, n_steps: usize) -> Result<(Vec<usize>, Vec<usize>), String> {
        let a = self.a.as_ref().ok_or("Transition matrix not set")?;
        let b = self.b.as_ref().ok_or("Emission matrix not set")?;
        let pi = self.pi.as_ref().ok_or("Initial state distribution not set")?;
        let _n = self.n.ok_or("Number of states not set")?;

        let mut rng = rand::thread_rng();

        let initial = WeightedIndex::new(pi.iter().cloned()).map_err(|e| e.to_string())?;
        let mut s = initial.sample(&mut rng);
        let mut states = vec![s];
        let mut emissions = vec![WeightedIndex::new(b.row(s).iter().cloned()).map_err(|e| e.to_string())?.sample(&mut rng)];

        for _ in 1..n_steps {
            s = WeightedIndex::new(a.row(s).iter().cloned()).map_err(|e| e.to_string())?.sample(&mut rng);
            states.push(s);
            emissions.push(WeightedIndex::new(b.row(s).iter().cloned()).map_err(|e| e.to_string())?.sample(&mut rng));
        }

        Ok((states, emissions))
    }

    /// Compute the log likelihood of a single observation sequence via the forward algorithm.
    pub fn log_likelihood(&self, o: &Array2<usize>) -> Result<f64, String> {
        if o.nrows() != 1 {
            return Err("Likelihood only accepts a single sequence".to_string());
        }
        let obs = o.row(0).to_owned();
        let forward = self.forward(&obs)?;
        let t = obs.len();
        let last: Vec<f64> = forward.column(t - 1).iter().cloned().collect();
        Ok(crate::numpy_ml::utils::logsumexp(&last))
    }

    /// Decode the most likely latent state sequence for an observation sequence using Viterbi.
    pub fn decode(&self, o: &Array2<usize>) -> Result<(Vec<usize>, f64), String> {
        if o.nrows() != 1 {
            return Err("Can only decode a single sequence".to_string());
        }

        let n = self.n.ok_or("Number of states not set")?;
        let a = self.a.as_ref().ok_or("Transition matrix not set")?;
        let b = self.b.as_ref().ok_or("Emission matrix not set")?;
        let pi = self.pi.as_ref().ok_or("Initial state distribution not set")?;
        let eps = self.eps;

        let obs = o.row(0).to_owned();
        let t = obs.len();

        let mut viterbi = Array2::<f64>::zeros((n, t));
        let mut back_pointer = Array2::<usize>::zeros((n, t));

        let o0 = obs[0];
        for s in 0..n {
            back_pointer[[s, 0]] = 0;
            viterbi[[s, 0]] = (pi[s] + eps).ln() + (b[[s, o0]] + eps).ln();
        }

        for tt in 1..t {
            let ot = obs[tt];
            for s in 0..n {
                let mut best_log_prob = f64::NEG_INFINITY;
                let mut best_prev = 0;
                for s_ in 0..n {
                    let log_prob = viterbi[[s_, tt - 1]] + (a[[s_, s]] + eps).ln() + (b[[s, ot]] + eps).ln();
                    if log_prob > best_log_prob {
                        best_log_prob = log_prob;
                        best_prev = s_;
                    }
                }
                viterbi[[s, tt]] = best_log_prob;
                back_pointer[[s, tt]] = best_prev;
            }
        }

        let mut best_path_log_prob = f64::NEG_INFINITY;
        let mut pointer = 0;
        for s in 0..n {
            if viterbi[[s, t - 1]] > best_path_log_prob {
                best_path_log_prob = viterbi[[s, t - 1]];
                pointer = s;
            }
        }

        let mut best_path = vec![pointer];
        for tt in (1..t).rev() {
            pointer = back_pointer[[pointer, tt]];
            best_path.push(pointer);
        }
        best_path.reverse();

        Ok((best_path, best_path_log_prob))
    }

    fn forward(&self, obs: &Array1<usize>) -> Result<Array2<f64>, String> {
        let n = self.n.ok_or("Number of states not set")?;
        let a = self.a.as_ref().ok_or("Transition matrix not set")?;
        let b = self.b.as_ref().ok_or("Emission matrix not set")?;
        let pi = self.pi.as_ref().ok_or("Initial state distribution not set")?;
        let eps = self.eps;

        let t = obs.len();
        let mut forward = Array2::<f64>::zeros((n, t));

        let o0 = obs[0];
        for s in 0..n {
            forward[[s, 0]] = (pi[s] + eps).ln() + (b[[s, o0]] + eps).ln();
        }

        for tt in 1..t {
            let ot = obs[tt];
            for s in 0..n {
                let mut vals = Vec::with_capacity(n);
                for s_ in 0..n {
                    vals.push(forward[[s_, tt - 1]] + (a[[s_, s]] + eps).ln() + (b[[s, ot]] + eps).ln());
                }
                forward[[s, tt]] = crate::numpy_ml::utils::logsumexp(&vals);
            }
        }

        Ok(forward)
    }

    fn backward(&self, obs: &Array1<usize>) -> Result<Array2<f64>, String> {
        let n = self.n.ok_or("Number of states not set")?;
        let a = self.a.as_ref().ok_or("Transition matrix not set")?;
        let b = self.b.as_ref().ok_or("Emission matrix not set")?;
        let eps = self.eps;

        let t = obs.len();
        let mut backward = Array2::<f64>::zeros((n, t));

        for s in 0..n {
            backward[[s, t - 1]] = 0.0;
        }

        for tt in (0..t - 1).rev() {
            let ot1 = obs[tt + 1];
            for s in 0..n {
                let mut vals = Vec::with_capacity(n);
                for s_ in 0..n {
                    vals.push((a[[s, s_]] + eps).ln() + (b[[s_, ot1]] + eps).ln() + backward[[s_, tt + 1]]);
                }
                backward[[s, tt]] = crate::numpy_ml::utils::logsumexp(&vals);
            }
        }

        Ok(backward)
    }

    fn initialize_parameters(&mut self) {
        let n = self.n.unwrap();
        let v = self.v.unwrap();
        let eps = self.eps;

        if self.pi.is_none() {
            self.pi = Some(Array1::from_elem(n, 1.0 / n as f64));
        }
        if self.a.is_none() {
            let mut a = Array2::from_elem((n, n), 1.0 / n as f64);
            a.mapv_inplace(|v| if v == 0.0 { eps } else { v });
            self.a = Some(a);
        }
        if self.b.is_none() {
            let mut rng = rand::thread_rng();
            let mut b = Array2::from_shape_fn((n, v), |_| rng.r#gen::<f64>());
            // Normalize rows
            for i in 0..n {
                let row_sum: f64 = b.row(i).iter().sum();
                for j in 0..v {
                    b[[i, j]] /= row_sum;
                }
            }
            self.b = Some(b);
        }
    }

    /// Fit the HMM parameters via the Baum-Welch (forward-backward) algorithm.
    pub fn fit(&mut self, o: &Array2<usize>, n_states: usize, n_observations: usize, pi: Option<Array1<f64>>, tol: f64, verbose: bool) -> Result<(), String> {
        let (i, _t) = o.dim();
        self.n = Some(n_states);
        self.v = Some(n_observations);
        if let Some(pi) = pi {
            self.pi = Some(pi);
        }
        self.initialize_parameters();

        let mut step = 0;
        let mut delta = f64::INFINITY;
        let mut ll_prev: f64 = (0..i).map(|idx| self.log_likelihood(&o.slice(ndarray::s![idx..idx+1, ..]).to_owned()).unwrap()).sum();

        while delta > tol {
            let (gamma, xi, phi) = self.e_step(o)?;
            self.m_step(o, &gamma, &xi, &phi)?;
            let ll: f64 = (0..i).map(|idx| self.log_likelihood(&o.slice(ndarray::s![idx..idx+1, ..]).to_owned()).unwrap()).sum();
            delta = ll - ll_prev;
            ll_prev = ll;
            step += 1;

            if verbose {
                println!("[Epoch {}] LL: {:.3} Delta: {:.5}", step, ll_prev, delta);
            }
        }

        Ok(())
    }

    fn e_step(&self, o: &Array2<usize>) -> Result<(Array3<f64>, Array4<f64>, Array2<f64>), String> {
        let (i, t) = o.dim();
        let n = self.n.ok_or("Number of states not set")?;
        let eps = self.eps;

        let mut phi = Array2::<f64>::zeros((i, n));
        let mut gamma = Array3::<f64>::zeros((i, n, t));
        let mut xi = Array4::<f64>::zeros((i, n, n, t));

        for idx in 0..i {
            let obs = o.row(idx).to_owned();
            let fwd = self.forward(&obs)?;
            let bwd = self.backward(&obs)?;
            let log_likelihood = crate::numpy_ml::utils::logsumexp(&fwd.column(t - 1).iter().cloned().collect::<Vec<_>>());

            let tt = t - 1;
            for si in 0..n {
                gamma[[idx, si, tt]] = fwd[[si, tt]] + bwd[[si, tt]] - log_likelihood;
                phi[[idx, si]] = fwd[[si, 0]] + bwd[[si, 0]] - log_likelihood;
            }

            for tt in 0..t - 1 {
                let ot1 = obs[tt + 1];
                let a = self.a.as_ref().unwrap();
                let b = self.b.as_ref().unwrap();
                for si in 0..n {
                    gamma[[idx, si, tt]] = fwd[[si, tt]] + bwd[[si, tt]] - log_likelihood;
                    for sj in 0..n {
                        xi[[idx, si, sj, tt]] = fwd[[si, tt]] + (a[[si, sj]] + eps).ln() + (b[[sj, ot1]] + eps).ln() + bwd[[sj, tt + 1]] - log_likelihood;
                    }
                }
            }
        }

        Ok((gamma, xi, phi))
    }

    fn m_step(&mut self, o: &Array2<usize>, gamma: &Array3<f64>, xi: &Array4<f64>, phi: &Array2<f64>) -> Result<(), String> {
        let (i, t) = o.dim();
        let n = self.n.ok_or("Number of states not set")?;
        let v = self.v.ok_or("Number of observations not set")?;
        let eps = self.eps;

        let mut a = Array2::<f64>::zeros((n, n));
        let mut b = Array2::<f64>::zeros((n, v));
        let mut pi = Array1::<f64>::zeros(n);

        let mut count_gamma = Array3::<f64>::zeros((i, n, v));
        let mut count_xi = Array3::<f64>::zeros((i, n, n));

        for idx in 0..i {
            let obs = o.row(idx).to_owned();
            for si in 0..n {
                for vk in 0..v {
                    let matching: Vec<usize> = obs.iter().enumerate().filter(|&(_, &val)| val == vk).map(|(pos, _)| pos).collect();
                    if matching.is_empty() {
                        count_gamma[[idx, si, vk]] = eps.ln();
                    } else {
                        let vals: Vec<f64> = matching.iter().map(|&pos| gamma[[idx, si, pos]]).collect();
                        count_gamma[[idx, si, vk]] = crate::numpy_ml::utils::logsumexp(&vals);
                    }
                }

                for sj in 0..n {
                    let vals: Vec<f64> = (0..t - 1).map(|tt| xi[[idx, si, sj, tt]]).collect();
                    count_xi[[idx, si, sj]] = crate::numpy_ml::utils::logsumexp(&vals);
                }
            }
        }

        let phi_vals: Vec<Vec<f64>> = (0..n).map(|si| (0..i).map(|idx| phi[[idx, si]]).collect()).collect();
        let pi_logsumexp: Vec<f64> = phi_vals.iter().map(|vals| crate::numpy_ml::utils::logsumexp(vals)).collect();
        for si in 0..n {
            pi[si] = (pi_logsumexp[si] - (i as f64 + eps).ln()).exp();
        }

        for si in 0..n {
            for vk in 0..v {
                let vals: Vec<f64> = (0..i).map(|idx| count_gamma[[idx, si, vk]]).collect();
                let numer = crate::numpy_ml::utils::logsumexp(&vals);
                let denom_vals: Vec<f64> = (0..v).map(|vkk| {
                    let vals2: Vec<f64> = (0..i).map(|idx| count_gamma[[idx, si, vkk]]).collect();
                    crate::numpy_ml::utils::logsumexp(&vals2)
                }).collect();
                let denom = crate::numpy_ml::utils::logsumexp(&denom_vals);
                b[[si, vk]] = (numer - denom).exp();
            }

            for sj in 0..n {
                let vals: Vec<f64> = (0..i).map(|idx| count_xi[[idx, si, sj]]).collect();
                let numer = crate::numpy_ml::utils::logsumexp(&vals);
                let denom_vals: Vec<f64> = (0..n).map(|sjj| {
                    let vals2: Vec<f64> = (0..i).map(|idx| count_xi[[idx, si, sjj]]).collect();
                    crate::numpy_ml::utils::logsumexp(&vals2)
                }).collect();
                let denom = crate::numpy_ml::utils::logsumexp(&denom_vals);
                a[[si, sj]] = (numer - denom).exp();
            }
        }

        self.a = Some(a);
        self.b = Some(b);
        self.pi = Some(pi);
        Ok(())
    }
}

/// A hidden Markov model with continuous Gaussian emission distributions.
pub struct GaussianHMM {
    pub a: Option<Array2<f64>>,
    pub means: Option<Array2<f64>>,
    pub covars: Option<Array2<f64>>,
    pub pi: Option<Array1<f64>>,
    pub n: Option<usize>,
    pub d: Option<usize>,
    pub eps: f64,
}

impl GaussianHMM {
    pub fn new(
        a: Option<Array2<f64>>,
        means: Option<Array2<f64>>,
        covars: Option<Array2<f64>>,
        pi: Option<Array1<f64>>,
        eps: Option<f64>,
    ) -> Self {
        let eps = eps.unwrap_or(1e-6);
        let n = a.as_ref().map(|a| a.nrows());
        let d = means.as_ref().map(|m| m.ncols());
        Self {
            a,
            means,
            covars,
            pi,
            n,
            d,
            eps,
        }
    }

    fn log_emission(&self, s: usize, o: &[f64]) -> f64 {
        let d = self.d.unwrap();
        let means = self.means.as_ref().unwrap();
        let covars = self.covars.as_ref().unwrap();
        let mut sum = -0.5 * (d as f64) * (2.0 * std::f64::consts::PI).ln();
        for k in 0..d {
            let var = covars[[s, k]].max(self.eps);
            let diff = o[k] - means[[s, k]];
            sum += -0.5 * (var.ln() + (diff * diff) / var);
        }
        sum
    }

    fn forward(&self, obs: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (t, _d) = obs.dim();
        let n = self.n.ok_or("Number of states not set")?;
        let a = self.a.as_ref().ok_or("Transition matrix not set")?;
        let pi = self.pi.as_ref().ok_or("Initial state distribution not set")?;
        let eps = self.eps;

        let mut fwd = Array2::<f64>::zeros((n, t));
        let o0: Vec<f64> = obs.row(0).to_vec();
        for s in 0..n {
            fwd[[s, 0]] = (pi[s] + eps).ln() + self.log_emission(s, &o0);
        }

        for tt in 1..t {
            let ot: Vec<f64> = obs.row(tt).to_vec();
            for s in 0..n {
                let log_emit = self.log_emission(s, &ot);
                let mut vals = Vec::with_capacity(n);
                for s_ in 0..n {
                    vals.push(fwd[[s_, tt - 1]] + (a[[s_, s]] + eps).ln() + log_emit);
                }
                fwd[[s, tt]] = crate::numpy_ml::utils::logsumexp(&vals);
            }
        }

        Ok(fwd)
    }

    fn backward(&self, obs: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (t, _d) = obs.dim();
        let n = self.n.ok_or("Number of states not set")?;
        let a = self.a.as_ref().ok_or("Transition matrix not set")?;
        let eps = self.eps;

        let mut bwd = Array2::<f64>::zeros((n, t));
        for s in 0..n {
            bwd[[s, t - 1]] = 0.0;
        }

        for tt in (0..t - 1).rev() {
            let ot1: Vec<f64> = obs.row(tt + 1).to_vec();
            for s in 0..n {
                let mut vals = Vec::with_capacity(n);
                for s_ in 0..n {
                    let log_emit = self.log_emission(s_, &ot1);
                    vals.push((a[[s, s_]] + eps).ln() + log_emit + bwd[[s_, tt + 1]]);
                }
                bwd[[s, tt]] = crate::numpy_ml::utils::logsumexp(&vals);
            }
        }

        Ok(bwd)
    }

    /// Compute the log likelihood of an observation sequence.
    pub fn log_likelihood(&self, obs: &Array2<f64>) -> Result<f64, String> {
        let fwd = self.forward(obs)?;
        let t = obs.nrows();
        let last: Vec<f64> = fwd.column(t - 1).iter().cloned().collect();
        Ok(crate::numpy_ml::utils::logsumexp(&last))
    }

    /// Decode the most likely latent state sequence using Viterbi.
    pub fn decode(&self, obs: &Array2<f64>) -> Result<(Vec<usize>, f64), String> {
        let (t, _d) = obs.dim();
        let n = self.n.ok_or("Number of states not set")?;
        let a = self.a.as_ref().ok_or("Transition matrix not set")?;
        let pi = self.pi.as_ref().ok_or("Initial state distribution not set")?;
        let eps = self.eps;

        let mut viterbi = Array2::<f64>::zeros((n, t));
        let mut back_pointer = Array2::<usize>::zeros((n, t));

        let o0: Vec<f64> = obs.row(0).to_vec();
        for s in 0..n {
            viterbi[[s, 0]] = (pi[s] + eps).ln() + self.log_emission(s, &o0);
        }

        for tt in 1..t {
            let ot: Vec<f64> = obs.row(tt).to_vec();
            for s in 0..n {
                let log_emit = self.log_emission(s, &ot);
                let mut best_lp = f64::NEG_INFINITY;
                let mut best_prev = 0;
                for s_ in 0..n {
                    let lp = viterbi[[s_, tt - 1]] + (a[[s_, s]] + eps).ln() + log_emit;
                    if lp > best_lp {
                        best_lp = lp;
                        best_prev = s_;
                    }
                }
                viterbi[[s, tt]] = best_lp;
                back_pointer[[s, tt]] = best_prev;
            }
        }

        let mut best_path_lp = f64::NEG_INFINITY;
        let mut pointer = 0;
        for s in 0..n {
            if viterbi[[s, t - 1]] > best_path_lp {
                best_path_lp = viterbi[[s, t - 1]];
                pointer = s;
            }
        }

        let mut path = vec![pointer];
        for tt in (1..t).rev() {
            pointer = back_pointer[[pointer, tt]];
            path.push(pointer);
        }
        path.reverse();

        Ok((path, best_path_lp))
    }

    /// Fit Gaussian HMM parameters using Baum-Welch (EM).
    pub fn fit(
        &mut self,
        obs: &Array2<f64>,
        n_states: usize,
        max_iter: usize,
        tol: f64,
    ) -> Result<(), String> {
        let (t, d) = obs.dim();
        self.n = Some(n_states);
        self.d = Some(d);

        // Initialize parameters if missing
        if self.pi.is_none() {
            self.pi = Some(Array1::from_elem(n_states, 1.0 / n_states as f64));
        }
        if self.a.is_none() {
            let mut a = Array2::from_elem((n_states, n_states), 1.0 / n_states as f64);
            a.mapv_inplace(|v| if v == 0.0 { self.eps } else { v });
            self.a = Some(a);
        }
        if self.means.is_none() {
            let mut means = Array2::<f64>::zeros((n_states, d));
            for s in 0..n_states {
                let row_idx = (s * t / n_states).min(t - 1);
                for k in 0..d {
                    means[[s, k]] = obs[[row_idx, k]];
                }
            }
            self.means = Some(means);
        }
        if self.covars.is_none() {
            self.covars = Some(Array2::from_elem((n_states, d), 1.0));
        }

        let mut ll_prev = f64::NEG_INFINITY;
        let eps = self.eps;

        for _ in 0..max_iter {
            let fwd = self.forward(obs)?;
            let bwd = self.backward(obs)?;
            let ll = crate::numpy_ml::utils::logsumexp(&fwd.column(t - 1).iter().cloned().collect::<Vec<_>>());

            if (ll - ll_prev).abs() < tol {
                break;
            }
            ll_prev = ll;

            // Compute gamma and xi
            let mut gamma = Array2::<f64>::zeros((n_states, t));
            for s in 0..n_states {
                for tt in 0..t {
                    gamma[[s, tt]] = (fwd[[s, tt]] + bwd[[s, tt]] - ll).exp();
                }
            }

            let a = self.a.as_ref().unwrap();
            let mut xi = ndarray::Array3::<f64>::zeros((n_states, n_states, t - 1));
            for tt in 0..t - 1 {
                let ot1: Vec<f64> = obs.row(tt + 1).to_vec();
                for i in 0..n_states {
                    for j in 0..n_states {
                        let log_emit = self.log_emission(j, &ot1);
                        xi[[i, j, tt]] = (fwd[[i, tt]] + (a[[i, j]] + eps).ln() + log_emit + bwd[[j, tt + 1]] - ll).exp();
                    }
                }
            }

            // M-step updates
            let mut new_pi = Array1::<f64>::zeros(n_states);
            let gamma_0_sum: f64 = (0..n_states).map(|s| gamma[[s, 0]]).sum();
            for s in 0..n_states {
                new_pi[s] = (gamma[[s, 0]] / gamma_0_sum.max(eps)).max(eps);
            }

            let mut new_a = Array2::<f64>::zeros((n_states, n_states));
            for i in 0..n_states {
                let denom: f64 = (0..t - 1).map(|tt| gamma[[i, tt]]).sum();
                for j in 0..n_states {
                    let num: f64 = (0..t - 1).map(|tt| xi[[i, j, tt]]).sum();
                    new_a[[i, j]] = (num / denom.max(eps)).max(eps);
                }
                let row_sum: f64 = new_a.row(i).iter().sum();
                for j in 0..n_states {
                    new_a[[i, j]] /= row_sum.max(eps);
                }
            }

            let mut new_means = Array2::<f64>::zeros((n_states, d));
            let mut new_covars = Array2::<f64>::zeros((n_states, d));
            for s in 0..n_states {
                let gamma_sum: f64 = gamma.row(s).iter().sum();
                let denom = gamma_sum.max(eps);
                for k in 0..d {
                    let num: f64 = (0..t).map(|tt| gamma[[s, tt]] * obs[[tt, k]]).sum();
                    new_means[[s, k]] = num / denom;

                    let var_num: f64 = (0..t)
                        .map(|tt| {
                            let diff = obs[[tt, k]] - new_means[[s, k]];
                            gamma[[s, tt]] * diff * diff
                        })
                        .sum();
                    new_covars[[s, k]] = (var_num / denom).max(1e-4);
                }
            }

            self.pi = Some(new_pi);
            self.a = Some(new_a);
            self.means = Some(new_means);
            self.covars = Some(new_covars);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gaussian_hmm_decode() {
        let a = array![[0.9, 0.1], [0.1, 0.9]];
        let means = array![[0.0], [5.0]];
        let covars = array![[1.0], [1.0]];
        let pi = array![0.5, 0.5];

        let ghmm = GaussianHMM::new(Some(a), Some(means), Some(covars), Some(pi), None);
        let obs = array![[0.1], [0.0], [-0.1], [5.1], [4.9], [5.0]];
        let (path, lp) = ghmm.decode(&obs).unwrap();

        assert_eq!(path, vec![0, 0, 0, 1, 1, 1]);
        assert!(lp < 0.0);
    }
}

