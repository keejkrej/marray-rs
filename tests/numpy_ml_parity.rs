use marray::numpy_ml::bandits::{BanditPolicy, BernoulliBandit, EpsilonGreedy, ThompsonSampling, UCB1};
use marray::numpy_ml::factorization::{NMF, VanillaALS};
use marray::numpy_ml::gmm::GMM;
use marray::numpy_ml::hmm::MultinomialHMM;
use marray::numpy_ml::lda::LDA;
use marray::numpy_ml::linear_models::{LinearRegression, LogisticRegression, RidgeRegression};
use marray::numpy_ml::neural_nets::activations::{Activation, ReLU, Sigmoid, Tanh};
use marray::numpy_ml::neural_nets::layers::{BatchNorm, Conv1D, Dropout, Linear, MultiHeadAttention};
use marray::numpy_ml::neural_nets::losses::{CrossEntropy, Loss, SquaredError};
use marray::numpy_ml::neural_nets::models::BernoulliVAE;
use marray::numpy_ml::ngram::{AdditiveNGram, MLENGram};
use marray::numpy_ml::nonparametric::{GPRegression, KernelRegression, KNN};
use marray::numpy_ml::rl_models::{Environment, QLearningAgent, SARSAAgent};
use marray::numpy_ml::trees::{DecisionTree, GradientBoostedDecisionTree, RandomForest};
use ndarray::{array, Array2, Array3};

struct SimpleLineEnv {
    state: usize,
}

impl SimpleLineEnv {
    fn new() -> Self {
        Self { state: 0 }
    }
}

impl Environment for SimpleLineEnv {
    fn n_states(&self) -> usize {
        4
    }

    fn n_actions(&self) -> usize {
        2
    }

    fn reset(&mut self) -> usize {
        self.state = 0;
        0
    }

    fn step(&mut self, action: usize) -> (usize, f64, bool) {
        if action == 1 && self.state < 3 {
            self.state += 1;
        } else if action == 0 && self.state > 0 {
            self.state -= 1;
        }
        let done = self.state == 3;
        let reward = if done { 10.0 } else { -0.1 };
        (self.state, reward, done)
    }
}

#[test]
fn test_numpy_ml_neural_nets_parity() {
    // Activations
    let x_act = array![[-1.0, 0.0, 1.0]];
    let relu = ReLU::new();
    let relu_out = relu.fn_(&x_act);
    assert_eq!(relu_out, array![[0.0, 0.0, 1.0]]);

    let sig = Sigmoid::new();
    let sig_out = sig.fn_(&x_act);
    assert!((sig_out[[0, 1]] - 0.5).abs() < 1e-6);

    let tanh = Tanh::new();
    let tanh_out = tanh.fn_(&x_act);
    assert!(tanh_out[[0, 1]].abs() < 1e-6);

    // Losses
    let mse = SquaredError::new();
    let y_true = array![[1.0, 0.0]];
    let y_pred = array![[0.5, 0.5]];
    let l_mse = mse.loss(&y_true, &y_pred);
    assert!((l_mse - 0.125).abs() < 1e-6);

    let ce = CrossEntropy::new();
    let y_onehot = array![[1.0, 0.0]];
    let y_prob = array![[0.8, 0.2]];
    let l_ce = ce.loss(&y_onehot, &y_prob);
    assert!(l_ce > 0.0);

    // Layers: Linear, Conv1D, BatchNorm, Dropout, MultiHeadAttention
    let mut linear = Linear::new(2, Some("relu"), "glorot_uniform", None);
    let lin_in = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let lin_out = linear.forward(&lin_in, false);
    assert_eq!(lin_out.dim(), (2, 2));

    let mut conv = Conv1D::new(1, 4, 3, Some(1), Some(1));
    let conv_input = Array3::from_shape_vec((2, 6, 1), vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        2.0, 1.0, 0.0, 1.0, 2.0, 3.0,
    ]).unwrap();
    let conv_out = conv.forward(&conv_input, false);
    assert_eq!(conv_out.dim().0, 2);
    assert_eq!(conv_out.dim().2, 4);

    let mut bn = BatchNorm::new(4, Some(0.9), Some(1e-5));
    let bn_in = Array2::from_elem((4, 4), 1.0);
    let bn_out = bn.forward(&bn_in, false);
    assert_eq!(bn_out.dim(), (4, 4));

    let mut dropout = Dropout::new(0.2);
    let drop_in = Array2::from_elem((4, 4), 1.0);
    let drop_out = dropout.forward(&drop_in, false);
    assert_eq!(drop_out.dim(), (4, 4));

    let mut mha = MultiHeadAttention::new(16, 4);
    let q = Array2::zeros((5, 16));
    let k = Array2::zeros((5, 16));
    let v = Array2::zeros((5, 16));
    let mha_out = mha.forward(&q, &k, &v, None, false);
    assert_eq!(mha_out.dim(), (5, 16));

    // BernoulliVAE
    let mut vae = BernoulliVAE::new(32, 4, 16);
    let x_vae = Array2::zeros((4, 32));
    let recon = vae.forward(&x_vae, false);
    assert_eq!(recon.dim(), (4, 32));
}

#[test]
fn test_numpy_ml_trees_and_bandits_parity() {
    let x = array![
        [0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]
    ];
    let y = array![0.0, 0.0, 1.0, 1.0];

    // Decision Tree
    let mut dt = DecisionTree::new(true, Some(3), None, Some("gini"), Some(42));
    dt.fit(&x, &y);
    let dt_preds = dt.predict(&x);
    assert_eq!(dt_preds, y);

    // Random Forest
    let mut rf = RandomForest::new(5, Some(3), None, true, Some("entropy"), Some(42));
    rf.fit(&x, &y);
    let rf_preds = rf.predict(&x);
    assert_eq!(rf_preds, y);

    // GBDT
    let mut gbdt = GradientBoostedDecisionTree::new(5, Some(2), true, 0.3, Some("cross_entropy"));
    gbdt.fit(&x, &y);
    let gbdt_preds = gbdt.predict(&x);
    assert_eq!(gbdt_preds, y);

    // Bandits
    let b = BernoulliBandit::new(vec![0.2, 0.8, 0.5]);
    assert_eq!(b.n_arms, 3);
    assert_eq!(b.best_arm, 1);

    let mut eg = EpsilonGreedy::new(0.1, 0.0);
    let (r, arm) = eg.act(&b);
    assert!(r == 0.0 || r == 1.0);
    assert!(arm < 3);

    let mut ucb = UCB1::new(1.0, 0.5);
    let (_r2, arm2) = ucb.act(&b);
    assert!(arm2 < 3);

    let mut ts = ThompsonSampling::new();
    let (_r3, arm3) = ts.act(&b);
    assert!(arm3 < 3);
}

#[test]
fn test_numpy_ml_rl_and_probabilistic_models_parity() {
    // Q-Learning Agent
    let mut env = SimpleLineEnv::new();
    let mut q_agent = QLearningAgent::new(4, 2, 0.1, 0.9, 1.0, 0.95, 0.01);
    for _ in 0..50 {
        let mut s = env.reset();
        for _ in 0..20 {
            let a = q_agent.act(s, true);
            let (next_s, r, done) = env.step(a);
            q_agent.update(s, a, r, next_s, done);
            s = next_s;
            if done { break; }
        }
    }
    let action = q_agent.act(0, false);
    assert!(action < 2);

    // SARSA Agent
    let mut sarsa_agent = SARSAAgent::new(4, 2, 0.1, 0.9, 1.0, 0.95, 0.01);
    for _ in 0..50 {
        let mut s = env.reset();
        let mut a = sarsa_agent.act(s, true);
        for _ in 0..20 {
            let (next_s, r, done) = env.step(a);
            let next_a = sarsa_agent.act(next_s, true);
            sarsa_agent.update(s, a, r, next_s, next_a, done);
            s = next_s;
            a = next_a;
            if done { break; }
        }
    }
    let action_sarsa = sarsa_agent.act(0, false);
    assert!(action_sarsa < 2);

    // GMM
    let x_gmm = array![
        [0.0, 0.0], [0.1, 0.1], [0.0, 0.1],
        [5.0, 5.0], [5.1, 5.1], [5.0, 5.1]
    ];
    let mut gmm = GMM::new(2, Some(42));
    gmm.fit(&x_gmm, 100, 1e-3, false).unwrap();
    let gmm_preds = gmm.predict(&x_gmm, false).unwrap().into_dimensionality::<ndarray::Ix1>().unwrap();
    assert_eq!(gmm_preds[0], gmm_preds[1]);
    assert_eq!(gmm_preds[3], gmm_preds[4]);
    assert_ne!(gmm_preds[0], gmm_preds[3]);

    // LDA
    let corpus = vec![
        vec![0, 1, 0, 1],
        vec![2, 3, 2, 3],
    ];
    let mut lda = LDA::new(2);
    lda.train(corpus, false, 20, 0.1);
    assert!(lda.beta.is_some());

    // Multinomial HMM
    let a = array![[0.9, 0.1], [0.1, 0.9]];
    let b_mat = array![[0.9, 0.1], [0.1, 0.9]];
    let pi = array![0.9, 0.1];
    let hmm = MultinomialHMM::new(Some(a), Some(b_mat), Some(pi), None);
    let obs = Array2::from_shape_vec((1, 4), vec![0, 0, 1, 1]).unwrap();
    let (path, log_prob) = hmm.decode(&obs).unwrap();
    assert_eq!(path.len(), 4);
    assert!(log_prob < 0.0);

    // N-gram Language Model
    let corpus_ngram = vec![
        vec![0, 1, 2],
        vec![0, 1, 1],
        vec![0, 2, 1],
    ];
    let mut mle = MLENGram::new(2);
    mle.train(&corpus_ngram);
    let pp = mle.perplexity(&vec![0, 1, 2], 2).unwrap();
    assert!(pp.is_finite());

    let mut add = AdditiveNGram::new(2, 1.0);
    add.train(&corpus_ngram);
    let pp_add = add.perplexity(&vec![0, 1, 2], 2).unwrap();
    assert!(pp_add.is_finite());
}

#[test]
fn test_numpy_ml_linear_and_nonparametric_parity() {
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![2.0, 4.0, 6.0, 8.0];

    // LinearRegression
    let mut lr = LinearRegression::new(true);
    lr.fit(&x, &y).unwrap();
    let preds = lr.predict(&array![[5.0]]).unwrap();
    assert!((preds[0] - 10.0).abs() < 1e-4);

    // RidgeRegression
    let mut ridge = RidgeRegression::new(1.0, true);
    ridge.fit(&x, &y).unwrap();
    let preds_ridge = ridge.predict(&array![[5.0]]).unwrap();
    assert!((preds_ridge[0] - 9.4545455).abs() < 1e-4);

    // LogisticRegression
    let x_clf = array![[0.0], [0.5], [1.5], [2.0]];
    let y_clf = array![0.0, 0.0, 1.0, 1.0];
    let mut logreg = LogisticRegression::new("l2", 0.01, true);
    logreg.fit(&x_clf, &y_clf, 0.1, 1e-4, 500).unwrap();
    let preds_log = logreg.predict(&array![[0.1], [1.9]]).unwrap();
    assert_eq!(preds_log[0], 0.0);
    assert_eq!(preds_log[1], 1.0);

    // KNN
    let mut knn = KNN::new(1, true, "uniform");
    knn.fit(&x_clf, &y_clf);
    let knn_preds = knn.predict(&array![[0.1], [1.9]]);
    assert_eq!(knn_preds[0], 0.0);
    assert_eq!(knn_preds[1], 1.0);

    // Kernel Regression
    let mut kr = KernelRegression::new(Some("rbf")).unwrap();
    kr.fit(&x, &y);
    let kr_preds = kr.predict(&array![[2.0]]);
    assert!((kr_preds[0] - 4.0).abs() < 0.5);

    // GP Regression
    let mut gp = GPRegression::new(Some("rbf"), 1e-4).unwrap();
    gp.fit(&x, &y);
    let (gp_preds, _, _) = gp.predict(&array![[2.0]], false).unwrap();
    assert!((gp_preds[0] - 4.0).abs() < 0.1);

    // Matrix Factorization: VanillaALS and NMF
    let ratings = array![
        [5.0, 3.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 5.0, 4.0]
    ];
    let mut als = VanillaALS::new(2, 0.1, 20, 1e-4);
    als.fit(&ratings, None, None, 1, false).unwrap();
    let (w_als, h_als) = als.parameters();
    assert!(w_als.is_some() && h_als.is_some());

    let mut nmf = NMF::new(2, 20, 1e-4);
    nmf.fit(&ratings, None, None, 1, false).unwrap();
    let (w_nmf, h_nmf) = nmf.parameters();
    assert!(w_nmf.is_some() && h_nmf.is_some());
}
