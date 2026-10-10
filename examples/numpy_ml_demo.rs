use mlab::numpy_ml::bandits::{BanditPolicy, BernoulliBandit, EpsilonGreedy, UCB1};
use mlab::numpy_ml::neural_nets::layers::{BatchNorm, Conv1D, Dropout, Linear, MultiHeadAttention};
use mlab::numpy_ml::neural_nets::models::BernoulliVAE;
use mlab::numpy_ml::rl_models::{Environment, QLearningAgent, SARSAAgent};
use mlab::numpy_ml::trees::{DecisionTree, RandomForest};
use ndarray::{array, Array2, Array3};

struct SimpleGridWorld {
    state: usize,
}

impl SimpleGridWorld {
    fn new() -> Self {
        Self { state: 0 }
    }
}

impl Environment for SimpleGridWorld {
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

fn main() {
    println!("=== numpy-ml Rust Demo ===\n");

    // 1. Neural Network Layers
    println!("1. Neural Network Layers (Conv1D, Linear, BatchNorm, Dropout, MultiHeadAttention):");
    let mut conv = Conv1D::new(1, 4, 3, Some(1), Some(1));
    let mut linear = Linear::new(2, Some("relu"), "glorot_uniform", None);
    let mut bn = BatchNorm::new(4, Some(0.9), Some(1e-5));
    let mut dropout = Dropout::new(0.2);

    // Conv1D input shape: (batch_size, length, in_channels)
    let conv_input = Array3::from_shape_vec((2, 6, 1), vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        2.0, 1.0, 0.0, 1.0, 2.0, 3.0,
    ]).unwrap();

    let conv_out = conv.forward(&conv_input, false);
    println!("  Conv1D output shape: {:?}", conv_out.dim());

    let lin_in = Array2::from_shape_vec((2, 4), vec![
        1.0, 2.0, 3.0, 4.0,
        0.5, 1.5, 2.5, 3.5,
    ]).unwrap();
    let lin_out = linear.forward(&lin_in, false);
    println!("  Linear output shape: {:?}", lin_out.dim());

    let bn_out = bn.forward(&lin_in, false);
    println!("  BatchNorm output shape: {:?}", bn_out.dim());

    dropout.is_training = false;
    let drop_out = dropout.forward(&lin_in, false);
    println!("  Dropout (eval mode) shape: {:?}", drop_out.dim());

    let mut mha = MultiHeadAttention::new(8, 2);
    let seq = Array2::zeros((4, 8)); // 4 tokens of dim 8
    let mha_out = mha.forward(&seq, &seq, &seq, None, false);
    println!("  MultiHeadAttention output shape: {:?}\n", mha_out.dim());

    // 2. Generative Models (Variational Autoencoder)
    println!("2. Generative Models (BernoulliVAE):");
    let mut vae = BernoulliVAE::new(10, 4, 16);
    let vae_input = Array2::from_elem((3, 10), 0.5);
    let recon = vae.forward(&vae_input, true);
    println!("  Reconstruction shape: {:?}", recon.dim());
    if let (Some(mu), Some(logvar)) = (&vae.t_mean_cache, &vae.t_log_var_cache) {
        println!("  Latent mu shape: {:?}", mu.dim());
        println!("  Latent logvar shape: {:?}\n", logvar.dim());
    }

    // 3. Decision Trees and Random Forests
    println!("3. Decision Trees & Ensembles:");
    let x_train = array![
        [0.0, 0.0],
        [0.1, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.05, 0.15],
        [0.95, 1.05]
    ];
    let y_train = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let mut dt = DecisionTree::new(true, Some(3), None, Some("entropy"), Some(42));
    dt.fit(&x_train, &y_train);
    let dt_preds = dt.predict(&x_train);
    println!("  DecisionTree predictions: {:?}", dt_preds);

    let mut rf = RandomForest::new(5, Some(3), None, true, Some("entropy"), Some(42));
    rf.fit(&x_train, &y_train);
    let rf_preds = rf.predict(&x_train);
    println!("  RandomForest predictions: {:?}\n", rf_preds);

    // 4. Multi-Armed Bandits
    println!("4. Multi-Armed Bandits:");
    let bandit = BernoulliBandit::new(vec![0.1, 0.8, 0.3]);
    let mut eps_policy = EpsilonGreedy::new(0.1, 0.0);
    let mut ucb_policy = UCB1::new(2.0, 0.0);

    let mut eps_rewards = 0.0;
    let mut ucb_rewards = 0.0;
    for _ in 0..100 {
        let (r_eps, _) = eps_policy.act(&bandit);
        eps_rewards += r_eps;

        let (r_ucb, _) = ucb_policy.act(&bandit);
        ucb_rewards += r_ucb;
    }
    println!("  EpsilonGreedy total rewards over 100 pulls: {}", eps_rewards);
    println!("  UCB1 total rewards over 100 pulls: {}\n", ucb_rewards);

    // 5. Reinforcement Learning Agents
    println!("5. Reinforcement Learning (Q-Learning and SARSA):");
    let mut grid_env = SimpleGridWorld::new();
    let mut q_agent = QLearningAgent::new(grid_env.n_states(), grid_env.n_actions(), 0.1, 0.9, 0.1, 0.99, 0.01);
    let mut sarsa_agent = SARSAAgent::new(grid_env.n_states(), grid_env.n_actions(), 0.1, 0.9, 0.1, 0.99, 0.01);

    for _ in 0..50 {
        let mut s = grid_env.reset();
        let mut done = false;
        while !done {
            let a = q_agent.act(s, true);
            let (next_s, r, d) = grid_env.step(a);
            q_agent.update(s, a, r, next_s, d);
            s = next_s;
            done = d;
        }

        s = grid_env.reset();
        done = false;
        let mut a = sarsa_agent.act(s, true);
        while !done {
            let (next_s, r, d) = grid_env.step(a);
            let next_a = sarsa_agent.act(next_s, true);
            sarsa_agent.update(s, a, r, next_s, next_a, d);
            s = next_s;
            a = next_a;
            done = d;
        }
    }
    println!("  Q-Learning state-action values learned successfully.");
    println!("  SARSA state-action values learned successfully.");
    println!("\nAll numpy-ml demonstrations completed successfully!");
}
