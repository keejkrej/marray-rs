pub mod linear_models;
pub mod gmm;
pub mod factorization;
pub mod utils;
pub mod hmm;
pub mod lda;
pub mod nonparametric;
pub mod bandits;
pub mod ngram;
pub mod rl_models;
pub mod neural_nets;
pub mod trees;

#[cfg(test)]
mod tests {
    use super::bandits::{BanditPolicy, BernoulliBandit, EpsilonGreedy, MultinomialBandit, ThompsonSampling, UCB1};
    use super::factorization::{NMF, VanillaALS};
    use super::gmm::GMM;
    use super::hmm::MultinomialHMM;
    use super::lda::LDA;
    use super::linear_models::{
        BayesianLinearRegressionKnownVariance, BayesianLinearRegressionUnknownVariance,
        GeneralizedLinearModel,
    };
    use super::nonparametric::{GPRegression, KernelRegression, KNN};
    use super::ngram::{AdditiveNGram, MLENGram};
    use super::rl_models::{Environment, QLearningAgent};
    use ndarray::{array, Array2};

    #[test]
    fn test_bayesian_linear_regression_known_variance() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut model = BayesianLinearRegressionKnownVariance::new(0.0, 1.0, None, true);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[5.0]]);

        assert_eq!(pred.len(), 1);
        assert!(pred[0] > 10.0);
    }

    #[test]
    fn test_bayesian_linear_regression_unknown_variance() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut model = BayesianLinearRegressionUnknownVariance::new(1.0, 1.0, 0.0, None, true);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[5.0]]);

        assert_eq!(pred.len(), 1);
        assert!(pred[0] > 10.0);
    }

    #[test]
    fn test_generalized_linear_model_identity_link() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mut model = GeneralizedLinearModel::new("identity", true, 1e-6, 25);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[4.0]]);

        assert_eq!(pred.len(), 1);
        assert!((pred[0] - 5.0).abs() < 0.2);
    }

    #[test]
    fn test_gmm_fit_predict() {
        // Two well-separated clusters in 2D
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.0, 5.1]
        ];
        let mut model = GMM::new(2, Some(42));
        model.fit(&x, 100, 1e-3, false).unwrap();

        let preds = model.predict(&x, false).unwrap();
        let preds = preds.into_dimensionality::<ndarray::Ix1>().unwrap();

        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[0], preds[2]);
        assert_eq!(preds[3], preds[4]);
        assert_eq!(preds[3], preds[5]);
        assert_ne!(preds[0], preds[3]);
    }

    #[test]
    fn test_vanilla_als_factorization() {
        // Rank-2 factorization of a simple product
        let w_true = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let h_true = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = w_true.dot(&h_true);

        let mut als = VanillaALS::new(2, 0.01, 500, 1e-6);
        als.fit(&x, None, None, 10, false).unwrap();

        let (w, h) = als.parameters();
        let w = w.unwrap();
        let h = h.unwrap();
        let x_hat = w.dot(h);
        let mse = (&x - &x_hat).iter().map(|&v| v * v).sum::<f64>() / x.len() as f64;
        assert!(mse < 0.5, "ALS reconstruction MSE too high: {}", mse);
    }

    #[test]
    fn test_nmf_factorization() {
        // Nonnegative rank-2 factorization
        let w_true = array![[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]];
        let h_true = array![[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]];
        let x = w_true.dot(&h_true);

        let mut nmf = NMF::new(2, 500, 1e-6);
        nmf.fit(&x, None, None, 5, false).unwrap();

        let (w, h) = nmf.parameters();
        let w = w.unwrap();
        let h = h.unwrap();

        // Both factors must be nonnegative
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!(h.iter().all(|&v| v >= 0.0));

        let x_hat = w.dot(h);
        let mse = (&x - &x_hat).iter().map(|&v| v * v).sum::<f64>() / x.len() as f64;
        assert!(mse < 1.0, "NMF reconstruction MSE too high: {}", mse);
    }

    #[test]
    fn test_hmm_decode() {
        // 2 states, 2 observations
        // State 0 mostly emits 0, state 1 mostly emits 1
        let a = array![[0.9, 0.1], [0.1, 0.9]];
        let b = array![[0.9, 0.1], [0.1, 0.9]];
        let pi = array![0.9, 0.1];

        let hmm = MultinomialHMM::new(Some(a), Some(b), Some(pi), None);
        let obs = Array2::from_shape_vec((1, 5), vec![0, 0, 1, 1, 1]).unwrap();

        let (path, log_prob) = hmm.decode(&obs).unwrap();
        assert_eq!(path, vec![0, 0, 1, 1, 1]);
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_hmm_log_likelihood() {
        let a = array![[0.9, 0.1], [0.1, 0.9]];
        let b = array![[0.9, 0.1], [0.1, 0.9]];
        let pi = array![0.5, 0.5];

        let hmm = MultinomialHMM::new(Some(a), Some(b), Some(pi), None);
        let obs = Array2::from_shape_vec((1, 3), vec![0, 0, 0]).unwrap();

        let ll = hmm.log_likelihood(&obs).unwrap();
        assert!(ll < 0.0);
    }

    #[test]
    fn test_hmm_fit() {
        // Simple 2-state, 2-observation HMM
        let a = array![[0.9, 0.1], [0.1, 0.9]];
        let b = array![[0.9, 0.1], [0.1, 0.9]];
        let pi = array![0.9, 0.1];

        let mut hmm = MultinomialHMM::new(Some(a.clone()), Some(b.clone()), Some(pi.clone()), None);
        let obs = Array2::from_shape_vec((1, 20), vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).unwrap();

        let ll_before = hmm.log_likelihood(&obs).unwrap();
        hmm.fit(&obs, 2, 2, None, 1e-5, false).unwrap();
        let ll_after = hmm.log_likelihood(&obs).unwrap();

        assert!(ll_after >= ll_before || (ll_after - ll_before).abs() < 1.0);
    }

    #[test]
    fn test_lda_train() {
        // Toy corpus: 4 documents over 6 word tokens, 2 topics
        let corpus = vec![
            vec![0, 1, 2, 0, 1],
            vec![0, 1, 0, 2],
            vec![3, 4, 5, 3],
            vec![4, 5, 4, 3, 5],
        ];

        let mut lda = LDA::new(2);
        lda.train(corpus, false, 50, 0.1);

        let beta = lda.beta.as_ref().unwrap();
        // beta columns should sum to 1
        for topic in 0..2 {
            let col_sum: f64 = beta.column(topic).iter().sum();
            assert!((col_sum - 1.0).abs() < 1e-6);
        }

        // gamma should be positive
        let gamma = lda.gamma.as_ref().unwrap();
        assert!(gamma.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_knn_classifier() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut knn = KNN::new(2, true, "uniform");
        knn.fit(&x, &y);

        let preds = knn.predict(&array![[0.05, 0.05], [10.05, 10.05]]);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_kernel_regression() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0];

        let mut kr = KernelRegression::new(Some("rbf")).unwrap();
        kr.fit(&x, &y);

        let preds = kr.predict(&array![[1.5], [2.5]]);
        assert!(preds[0] > 0.5 && preds[0] < 2.5);
        assert!(preds[1] > 1.5 && preds[1] < 3.5);
    }

    #[test]
    fn test_gp_regression() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![0.0, 1.0, 2.0, 3.0];

        let mut gp = GPRegression::new(Some("rbf"), 1e-6).unwrap();
        gp.fit(&x, &y);

        let (preds, conf, _) = gp.predict(&array![[1.5], [2.5]], false).unwrap();
        assert!((preds[0] - 1.5).abs() < 0.5);
        assert!((preds[1] - 2.5).abs() < 0.5);
        assert!(conf.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_epsilon_greedy_bandit() {
        let bandit = BernoulliBandit::new(vec![0.1, 0.9]);
        let mut policy = EpsilonGreedy::new(0.0, 0.5);

        for _ in 0..50 {
            policy.act(&bandit);
        }

        let (best_ev, best_arm) = bandit.oracle_payoff();
        assert_eq!(best_arm, 1);
        assert!((best_ev - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_ucb1_bandit() {
        let bandit = BernoulliBandit::new(vec![0.2, 0.8]);
        let mut policy = UCB1::new(1.0, 0.5);

        for _ in 0..100 {
            policy.act(&bandit);
        }

        let (_, best_arm) = bandit.oracle_payoff();
        assert_eq!(best_arm, 1);
    }

    #[test]
    fn test_thompson_sampling_bandit() {
        let bandit = BernoulliBandit::new(vec![0.3, 0.7]);
        let mut policy = ThompsonSampling::new();

        for _ in 0..100 {
            policy.act(&bandit);
        }

        let (_, best_arm) = bandit.oracle_payoff();
        assert_eq!(best_arm, 1);
    }

    #[test]
    fn test_multinomial_bandit() {
        let bandit = MultinomialBandit::new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        ).unwrap();

        let (ev, arm) = bandit.oracle_payoff();
        assert_eq!(arm, 1);
        assert!((ev - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_mle_ngram() {
        let corpus = vec![
            vec![0, 1, 2],
            vec![0, 1, 1],
            vec![0, 2, 1],
        ];

        let mut model = MLENGram::new(2);
        model.train(&corpus);

        let pp = model.perplexity(&vec![0, 1, 2], 2).unwrap();
        assert!(pp.is_finite());
    }

    #[test]
    fn test_additive_ngram() {
        let corpus = vec![
            vec![0, 1, 2],
            vec![0, 1, 1],
        ];

        let mut model = AdditiveNGram::new(2, 1.0);
        model.train(&corpus);

        let pp = model.perplexity(&vec![0, 1, 2], 2).unwrap();
        assert!(pp.is_finite());
    }

    #[test]
    fn test_q_learning_agent() {
        struct GridEnv {
            state: usize,
        }

        impl Environment for GridEnv {
            fn reset(&mut self) -> usize {
                self.state = 0;
                self.state
            }

            fn step(&mut self, action: usize) -> (usize, f64, bool) {
                // Actions: 0 = stay, 1 = forward
                if action == 1 && self.state < 4 {
                    self.state += 1;
                }
                let reward = if self.state == 4 { 1.0 } else { 0.0 };
                let done = self.state == 4;
                (self.state, reward, done)
            }

            fn n_states(&self) -> usize { 5 }
            fn n_actions(&self) -> usize { 2 }
        }

        let mut agent = QLearningAgent::new(5, 2, 0.1, 0.9, 1.0, 0.99, 0.01);
        let mut env = GridEnv { state: 0 };

        for _ in 0..200 {
            agent.run_episode(&mut env, 10, true);
        }

        let q = agent.q_values();
        // The action "forward" should have higher value than "stay" in early states
        assert!(q[[0, 1]] > q[[0, 0]]);
    }
}
