use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Distribution;

/// A minimal reinforcement-learning environment trait.
pub trait Environment {
    /// Reset the environment and return the initial state index.
    fn reset(&mut self) -> usize;
    /// Take action `action` and return (next_state, reward, done).
    fn step(&mut self, action: usize) -> (usize, f64, bool);
    /// Number of discrete states.
    fn n_states(&self) -> usize;
    /// Number of discrete actions.
    fn n_actions(&self) -> usize;
}

/// Q-learning agent for discrete state/action environments.
pub struct QLearningAgent {
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
    q_values: Array2<f64>,
}

impl QLearningAgent {
    /// Create a new Q-learning agent.
    pub fn new(n_states: usize, n_actions: usize, alpha: f64, gamma: f64, epsilon: f64, epsilon_decay: f64, epsilon_min: f64) -> Self {
        Self {
            alpha,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            q_values: Array2::zeros((n_states, n_actions)),
        }
    }

    /// Select an action using epsilon-greedy.
    pub fn act(&self, state: usize, explore: bool) -> usize {
        let mut rng = rand::thread_rng();
        if explore && rng.r#gen::<f64>() < self.epsilon {
            rng.r#gen_range(0..self.q_values.ncols())
        } else {
            let mut best_action = 0;
            let mut best_value = self.q_values[[state, 0]];
            for a in 1..self.q_values.ncols() {
                if self.q_values[[state, a]] > best_value {
                    best_value = self.q_values[[state, a]];
                    best_action = a;
                }
            }
            best_action
        }
    }

    /// Update Q-value from a transition.
    pub fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize, done: bool) {
        let current = self.q_values[[state, action]];
        let target = if done {
            reward
        } else {
            let max_next = (0..self.q_values.ncols())
                .map(|a| self.q_values[[next_state, a]])
                .fold(f64::NEG_INFINITY, f64::max);
            reward + self.gamma * max_next
        };
        self.q_values[[state, action]] += self.alpha * (target - current);
    }

    /// Decay epsilon.
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    /// Run a single training episode and return total reward.
    pub fn run_episode(&mut self, env: &mut dyn Environment, max_steps: usize, explore: bool) -> f64 {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        for _ in 0..max_steps {
            let action = self.act(state, explore);
            let (next_state, reward, done) = env.step(action);
            if explore {
                self.update(state, action, reward, next_state, done);
            }
            total_reward += reward;
            state = next_state;
            if done {
                break;
            }
        }
        if explore {
            self.decay_epsilon();
        }
        total_reward
    }

    /// Return a reference to the learned Q-values.
    pub fn q_values(&self) -> &Array2<f64> {
        &self.q_values
    }
}

/// A simple tabular cross-entropy method agent.
pub struct CrossEntropyAgent {
    pub n_samples: usize,
    pub retain_prct: f64,
    theta_mean: Array1<f64>,
    theta_std: Array1<f64>,
    n_actions: usize,
}

impl CrossEntropyAgent {
    /// Create a new cross-entropy agent.
    pub fn new(n_states: usize, n_actions: usize, n_samples: usize, retain_prct: f64) -> Self {
        let theta_dim = n_states * n_actions;
        Self {
            n_samples,
            retain_prct,
            theta_mean: Array1::from_elem(theta_dim, 0.0),
            theta_std: Array1::ones(theta_dim),
            n_actions,
        }
    }

    fn sample_theta(&self) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        Array1::from_shape_fn(self.theta_mean.len(), |i| {
            let normal = rand_distr::Normal::new(self.theta_mean[i], self.theta_std[i]).unwrap();
            normal.sample(&mut rng)
        })
    }

    fn act_with_theta(&self, state: usize, theta: &Array1<f64>) -> usize {
        let mut best_action = 0;
        let mut best_value = theta[state * self.n_actions];
        for a in 1..self.n_actions {
            let v = theta[state * self.n_actions + a];
            if v > best_value {
                best_value = v;
                best_action = a;
            }
        }
        best_action
    }

    fn evaluate_theta(&self, env: &mut dyn Environment, theta: &Array1<f64>, max_steps: usize) -> f64 {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        for _ in 0..max_steps {
            let action = self.act_with_theta(state, theta);
            let (next_state, reward, done) = env.step(action);
            total_reward += reward;
            state = next_state;
            if done {
                break;
            }
        }
        total_reward
    }

    /// Run one iteration of cross-entropy training.
    pub fn run_iteration(&mut self, env: &mut dyn Environment, max_steps: usize) -> f64 {
        let mut samples: Vec<(Array1<f64>, f64)> = (0..self.n_samples)
            .map(|_| {
                let theta = self.sample_theta();
                let reward = self.evaluate_theta(env, &theta, max_steps);
                (theta, reward)
            })
            .collect();

        samples.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_retain = (self.n_samples as f64 * self.retain_prct).ceil() as usize;
        let retained: Vec<&Array1<f64>> = samples.iter().take(n_retain).map(|(theta, _)| theta).collect();

        for i in 0..self.theta_mean.len() {
            let vals: Vec<f64> = retained.iter().map(|t| t[i]).collect();
            self.theta_mean[i] = vals.iter().sum::<f64>() / vals.len() as f64;
            let mean = self.theta_mean[i];
            let var: f64 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            self.theta_std[i] = var.sqrt().max(1e-6);
        }

        samples.iter().map(|(_, r)| r).sum::<f64>() / samples.len() as f64
    }
}

/// On-policy SARSA agent for discrete state/action environments.
pub struct SARSAAgent {
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
    q_values: Array2<f64>,
}

impl SARSAAgent {
    /// Create a new SARSA agent.
    pub fn new(
        n_states: usize,
        n_actions: usize,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
        epsilon_decay: f64,
        epsilon_min: f64,
    ) -> Self {
        Self {
            alpha,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            q_values: Array2::zeros((n_states, n_actions)),
        }
    }

    /// Select an action using epsilon-greedy.
    pub fn act(&self, state: usize, explore: bool) -> usize {
        let mut rng = rand::thread_rng();
        if explore && rng.r#gen::<f64>() < self.epsilon {
            rng.r#gen_range(0..self.q_values.ncols())
        } else {
            let mut best_action = 0;
            let mut best_value = self.q_values[[state, 0]];
            for a in 1..self.q_values.ncols() {
                if self.q_values[[state, a]] > best_value {
                    best_value = self.q_values[[state, a]];
                    best_action = a;
                }
            }
            best_action
        }
    }

    /// Update Q-value from transition (s, a, r, s', a').
    pub fn update(
        &mut self,
        state: usize,
        action: usize,
        reward: f64,
        next_state: usize,
        next_action: usize,
        done: bool,
    ) {
        let current = self.q_values[[state, action]];
        let target = if done {
            reward
        } else {
            reward + self.gamma * self.q_values[[next_state, next_action]]
        };
        self.q_values[[state, action]] += self.alpha * (target - current);
    }

    /// Decay epsilon.
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    /// Run a single training episode using SARSA and return total reward.
    pub fn run_episode(&mut self, env: &mut dyn Environment, max_steps: usize, explore: bool) -> f64 {
        let mut state = env.reset();
        let mut action = self.act(state, explore);
        let mut total_reward = 0.0;

        for _ in 0..max_steps {
            let (next_state, reward, done) = env.step(action);
            total_reward += reward;

            if done {
                if explore {
                    self.update(state, action, reward, next_state, 0, true);
                }
                break;
            }

            let next_action = self.act(next_state, explore);
            if explore {
                self.update(state, action, reward, next_state, next_action, false);
            }
            state = next_state;
            action = next_action;
        }

        if explore {
            self.decay_epsilon();
        }
        total_reward
    }

    /// Return a reference to the learned Q-values.
    pub fn q_values(&self) -> &Array2<f64> {
        &self.q_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleLineEnv {
        state: usize,
    }

    impl Environment for SimpleLineEnv {
        fn reset(&mut self) -> usize {
            self.state = 0;
            0
        }

        fn step(&mut self, action: usize) -> (usize, f64, bool) {
            if action == 1 && self.state < 3 {
                self.state += 1;
            }
            let done = self.state == 3;
            let reward = if done { 10.0 } else { 0.0 };
            (self.state, reward, done)
        }

        fn n_states(&self) -> usize {
            4
        }
        fn n_actions(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_sarsa_agent() {
        let mut env = SimpleLineEnv { state: 0 };
        let mut agent = SARSAAgent::new(4, 2, 0.2, 0.9, 0.5, 0.99, 0.01);

        for _ in 0..300 {
            agent.run_episode(&mut env, 20, true);
        }

        let q = agent.q_values();
        assert!(q[[0, 1]] > q[[0, 0]]);
        assert!(q[[1, 1]] > q[[1, 0]]);
    }
}

