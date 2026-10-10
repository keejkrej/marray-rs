use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use std::collections::HashMap;

/// A multi-armed bandit with Bernoulli-distributed arm payoffs.
pub struct BernoulliBandit {
    pub payoff_probs: Vec<f64>,
    pub n_arms: usize,
    pub best_arm: usize,
    pub best_ev: f64,
}

impl BernoulliBandit {
    /// Create a new Bernoulli bandit.
    pub fn new(payoff_probs: Vec<f64>) -> Self {
        let n_arms = payoff_probs.len();
        let (best_ev, best_arm) = payoff_probs
            .iter()
            .enumerate()
            .fold((0.0, 0), |(best_val, best_idx), (idx, &p)| {
                if p > best_val { (p, idx) } else { (best_val, best_idx) }
            });
        Self { payoff_probs, n_arms, best_arm, best_ev }
    }

    /// Pull an arm and return the reward.
    pub fn pull(&self, arm_id: usize) -> f64 {
        let mut rng = rand::thread_rng();
        if rng.r#gen::<f64>() <= self.payoff_probs[arm_id] { 1.0 } else { 0.0 }
    }

    /// Expected reward for an optimal agent.
    pub fn oracle_payoff(&self) -> (f64, usize) {
        (self.best_ev, self.best_arm)
    }
}

/// A multi-armed bandit with multinomial-distributed arm payoffs.
pub struct MultinomialBandit {
    pub payoffs: Vec<Vec<f64>>,
    pub payoff_probs: Vec<Vec<f64>>,
    pub n_arms: usize,
    pub best_arm: usize,
    pub best_ev: f64,
}

impl MultinomialBandit {
    /// Create a new multinomial bandit.
    pub fn new(payoffs: Vec<Vec<f64>>, payoff_probs: Vec<Vec<f64>>) -> Result<Self, String> {
        if payoffs.len() != payoff_probs.len() {
            return Err("payoffs and payoff_probs must have same length".to_string());
        }
        let n_arms = payoffs.len();
        let mut arm_evs = Vec::with_capacity(n_arms);
        for (p, pp) in payoffs.iter().zip(payoff_probs.iter()) {
            if p.len() != pp.len() {
                return Err("Each arm's payoffs and probs must match".to_string());
            }
            let sum: f64 = pp.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err("Payoff probabilities must sum to 1".to_string());
            }
            arm_evs.push(p.iter().zip(pp.iter()).map(|(&v, &p)| v * p).sum());
        }
        let (best_ev, best_arm) = arm_evs.iter().enumerate().fold((0.0, 0), |(bv, bi), (i, &v)| {
            if v > bv { (v, i) } else { (bv, bi) }
        });
        Ok(Self { payoffs, payoff_probs, n_arms, best_arm, best_ev })
    }

    /// Pull an arm and return the reward.
    pub fn pull(&self, arm_id: usize) -> f64 {
        let dist = WeightedIndex::new(&self.payoff_probs[arm_id]).unwrap();
        let mut rng = rand::thread_rng();
        let idx = dist.sample(&mut rng);
        self.payoffs[arm_id][idx]
    }

    /// Expected reward for an optimal agent.
    pub fn oracle_payoff(&self) -> (f64, usize) {
        (self.best_ev, self.best_arm)
    }
}

/// Base trait for bandit policies.
pub trait BanditPolicy {
    /// Select an arm, pull it, and update internal estimates.
    fn act(&mut self, bandit: &dyn Bandit) -> (f64, usize);
    /// Reset the policy.
    fn reset(&mut self);
}

/// A bandit exposes its number of arms and a pull method.
pub trait Bandit {
    fn n_arms(&self) -> usize;
    fn pull(&self, arm_id: usize) -> f64;
}

impl Bandit for BernoulliBandit {
    fn n_arms(&self) -> usize { self.n_arms }
    fn pull(&self, arm_id: usize) -> f64 { self.pull(arm_id) }
}

impl Bandit for MultinomialBandit {
    fn n_arms(&self) -> usize { self.n_arms }
    fn pull(&self, arm_id: usize) -> f64 { self.pull(arm_id) }
}

/// Epsilon-greedy bandit policy.
pub struct EpsilonGreedy {
    epsilon: f64,
    ev_prior: f64,
    ev_estimates: Vec<f64>,
    pull_counts: Vec<usize>,
    initialized: bool,
}

impl EpsilonGreedy {
    /// Create a new epsilon-greedy policy.
    pub fn new(epsilon: f64, ev_prior: f64) -> Self {
        Self {
            epsilon,
            ev_prior,
            ev_estimates: Vec::new(),
            pull_counts: Vec::new(),
            initialized: false,
        }
    }

    fn initialize(&mut self, n_arms: usize) {
        self.ev_estimates = vec![self.ev_prior; n_arms];
        self.pull_counts = vec![0; n_arms];
        self.initialized = true;
    }
}

impl BanditPolicy for EpsilonGreedy {
    fn act(&mut self, bandit: &dyn Bandit) -> (f64, usize) {
        if !self.initialized {
            self.initialize(bandit.n_arms());
        }

        let mut rng = rand::thread_rng();
        let arm_id = if rng.r#gen::<f64>() < self.epsilon {
            rng.r#gen_range(0..bandit.n_arms())
        } else {
            let (idx, _) = self.ev_estimates.iter().enumerate().fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
            idx
        };

        let reward = bandit.pull(arm_id);
        self.pull_counts[arm_id] += 1;
        self.ev_estimates[arm_id] += (reward - self.ev_estimates[arm_id]) / self.pull_counts[arm_id] as f64;
        (reward, arm_id)
    }

    fn reset(&mut self) {
        self.ev_estimates.clear();
        self.pull_counts.clear();
        self.initialized = false;
    }
}

/// UCB1 bandit policy.
pub struct UCB1 {
    c: f64,
    ev_prior: f64,
    ev_estimates: Vec<f64>,
    pull_counts: Vec<usize>,
    step: usize,
    initialized: bool,
}

impl UCB1 {
    /// Create a new UCB1 policy.
    pub fn new(c: f64, ev_prior: f64) -> Self {
        Self {
            c,
            ev_prior,
            ev_estimates: Vec::new(),
            pull_counts: Vec::new(),
            step: 0,
            initialized: false,
        }
    }

    fn initialize(&mut self, n_arms: usize) {
        self.ev_estimates = vec![self.ev_prior; n_arms];
        self.pull_counts = vec![0; n_arms];
        self.step = 0;
        self.initialized = true;
    }
}

impl BanditPolicy for UCB1 {
    fn act(&mut self, bandit: &dyn Bandit) -> (f64, usize) {
        if !self.initialized {
            self.initialize(bandit.n_arms());
        }
        self.step += 1;

        let arm_id = if self.step <= self.ev_estimates.len() {
            self.step - 1
        } else {
            let mut best_ucb = f64::NEG_INFINITY;
            let mut best_arm = 0;
            for i in 0..self.ev_estimates.len() {
                let n = self.pull_counts[i].max(1) as f64;
                let ucb = self.ev_estimates[i] + self.c * ((2.0 * (self.step as f64).ln()) / n).sqrt();
                if ucb > best_ucb {
                    best_ucb = ucb;
                    best_arm = i;
                }
            }
            best_arm
        };

        let reward = bandit.pull(arm_id);
        self.pull_counts[arm_id] += 1;
        self.ev_estimates[arm_id] += (reward - self.ev_estimates[arm_id]) / self.pull_counts[arm_id] as f64;
        (reward, arm_id)
    }

    fn reset(&mut self) {
        self.ev_estimates.clear();
        self.pull_counts.clear();
        self.step = 0;
        self.initialized = false;
    }
}

/// Thompson sampling with Beta-Binomial updates.
pub struct ThompsonSampling {
    alpha: Vec<f64>,
    beta: Vec<f64>,
    initialized: bool,
}

impl ThompsonSampling {
    /// Create a new Thompson sampling policy with Beta(1,1) priors.
    pub fn new() -> Self {
        Self { alpha: Vec::new(), beta: Vec::new(), initialized: false }
    }

    fn initialize(&mut self, n_arms: usize) {
        self.alpha = vec![1.0; n_arms];
        self.beta = vec![1.0; n_arms];
        self.initialized = true;
    }
}

impl Default for ThompsonSampling {
    fn default() -> Self {
        Self::new()
    }
}

impl BanditPolicy for ThompsonSampling {
    fn act(&mut self, bandit: &dyn Bandit) -> (f64, usize) {
        if !self.initialized {
            self.initialize(bandit.n_arms());
        }

        let mut rng = rand::thread_rng();
        let mut best_sample = f64::NEG_INFINITY;
        let mut best_arm = 0;
        for i in 0..self.alpha.len() {
            let sample = rand_distr::Beta::new(self.alpha[i], self.beta[i]).unwrap().sample(&mut rng);
            if sample > best_sample {
                best_sample = sample;
                best_arm = i;
            }
        }

        let reward = bandit.pull(best_arm);
        self.alpha[best_arm] += reward;
        self.beta[best_arm] += 1.0 - reward;
        (reward, best_arm)
    }

    fn reset(&mut self) {
        self.alpha.clear();
        self.beta.clear();
        self.initialized = false;
    }
}

/// Softmax (Boltzmann) exploration policy for multi-armed bandits.
pub struct SoftmaxPolicy {
    tau: f64,
    ev_prior: f64,
    ev_estimates: Vec<f64>,
    pull_counts: Vec<usize>,
    initialized: bool,
}

impl SoftmaxPolicy {
    pub fn new(tau: f64, ev_prior: f64) -> Self {
        Self {
            tau,
            ev_prior,
            ev_estimates: Vec::new(),
            pull_counts: Vec::new(),
            initialized: false,
        }
    }

    fn initialize(&mut self, n_arms: usize) {
        self.ev_estimates = vec![self.ev_prior; n_arms];
        self.pull_counts = vec![0; n_arms];
        self.initialized = true;
    }
}

impl BanditPolicy for SoftmaxPolicy {
    fn act(&mut self, bandit: &dyn Bandit) -> (f64, usize) {
        if !self.initialized {
            self.initialize(bandit.n_arms());
        }

        let tau = self.tau.max(1e-8);
        let max_val = self.ev_estimates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = self
            .ev_estimates
            .iter()
            .map(|&v| ((v - max_val) / tau).exp())
            .collect();
        let dist = WeightedIndex::new(&exp_vals).unwrap();
        let mut rng = rand::thread_rng();
        let arm_id = dist.sample(&mut rng);

        let reward = bandit.pull(arm_id);
        self.pull_counts[arm_id] += 1;
        self.ev_estimates[arm_id] +=
            (reward - self.ev_estimates[arm_id]) / self.pull_counts[arm_id] as f64;
        (reward, arm_id)
    }

    fn reset(&mut self) {
        self.ev_estimates.clear();
        self.pull_counts.clear();
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_policy() {
        let bandit = BernoulliBandit::new(vec![0.1, 0.9]);
        let mut policy = SoftmaxPolicy::new(0.1, 0.5);

        for _ in 0..150 {
            policy.act(&bandit);
        }

        let (_, best_arm) = bandit.oracle_payoff();
        assert_eq!(best_arm, 1);
        assert!(policy.ev_estimates[1] > policy.ev_estimates[0]);
    }
}

