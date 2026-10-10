use ndarray::{Array1, Array2};
use rand::Rng;
use rand::seq::SliceRandom;

fn glorot_uniform(rows: usize, cols: usize) -> Array2<f64> {
    let b = (6.0 / (rows + cols) as f64).sqrt();
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-b..b))
}

fn relu(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|v| v.max(0.0))
}

fn relu_grad(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

fn tanh_act(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|v| v.tanh())
}

fn tanh_grad(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|v| {
        let t = v.tanh();
        1.0 - t * t
    })
}

fn logistic(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn logistic_grad(z: &Array2<f64>) -> Array2<f64> {
    let s = logistic(z);
    &s * &(1.0 - &s)
}

fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut output = x.clone();
    for i in 0..x.nrows() {
        let max_val = (0..x.ncols()).map(|j| x[[i, j]]).fold(f64::NEG_INFINITY, f64::max);
        let mut exp_sum = 0.0;
        for j in 0..x.ncols() {
            output[[i, j]] = (x[[i, j]] - max_val).exp();
            exp_sum += output[[i, j]];
        }
        for j in 0..x.ncols() {
            output[[i, j]] /= exp_sum;
        }
    }
    output
}

fn one_hot(y: &Array1<f64>, n_classes: usize) -> Array2<f64> {
    let n = y.len();
    let mut oh = Array2::zeros((n, n_classes));
    for i in 0..n {
        let c = y[i] as usize;
        oh[[i, c]] = 1.0;
    }
    oh
}

fn minibatch_indices(n_samples: usize, batch_size: usize) -> Vec<Vec<usize>> {
    let n_batches = (n_samples + batch_size - 1) / batch_size;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);
    let mut batches = Vec::with_capacity(n_batches);
    for i in 0..n_batches {
        let start = i * batch_size;
        let end = (start + batch_size).min(n_samples);
        batches.push(indices[start..end].to_vec());
    }
    batches
}

struct MLPNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    activation: String,
}

impl MLPNetwork {
    fn new(layer_sizes: &[usize], activation: &str) -> Self {
        let n_layers = layer_sizes.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let w = glorot_uniform(layer_sizes[i], layer_sizes[i + 1]);
            let b = Array1::zeros(layer_sizes[i + 1]);
            weights.push(w);
            biases.push(b);
        }
        MLPNetwork {
            weights,
            biases,
            activation: activation.to_string(),
        }
    }

    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activation.as_str() {
            "relu" => relu(z),
            "tanh" => tanh_act(z),
            "logistic" => logistic(z),
            _ => relu(z),
        }
    }

    fn activate_grad(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activation.as_str() {
            "relu" => relu_grad(z),
            "tanh" => tanh_grad(z),
            "logistic" => logistic_grad(z),
            _ => relu_grad(z),
        }
    }

    fn forward(&self, x: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let n_layers = self.weights.len();
        let mut pre_activations = Vec::with_capacity(n_layers);
        let mut activations = Vec::with_capacity(n_layers);

        let mut current = x.clone();
        let batch_size = x.nrows();
        for i in 0..n_layers {
            let z = current.dot(&self.weights[i])
                + &self.biases[i].broadcast((batch_size, self.biases[i].len())).unwrap().to_owned();
            pre_activations.push(z.clone());
            if i < n_layers - 1 {
                current = self.activate(&z);
            } else {
                current = z;
            }
            activations.push(current.clone());
        }

        (pre_activations, activations)
    }
}

pub struct MLPClassifier {
    pub hidden_layer_sizes: Vec<usize>,
    pub activation: String,
    pub learning_rate: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub batch_size: usize,
    pub alpha: f64,
    network: Option<MLPNetwork>,
    n_classes: usize,
}

impl MLPClassifier {
    pub fn new(
        hidden_layer_sizes: Vec<usize>,
        activation: &str,
        learning_rate: f64,
        max_iter: usize,
    ) -> Self {
        MLPClassifier {
            hidden_layer_sizes,
            activation: activation.to_string(),
            learning_rate,
            max_iter,
            tol: 1e-4,
            batch_size: 200,
            alpha: 0.0001,
            network: None,
            n_classes: 0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (n_samples, n_features) = x.dim();
        let classes: Vec<f64> = {
            let mut c: Vec<f64> = y.iter().copied().collect();
            c.sort_by(|a, b| a.partial_cmp(b).unwrap());
            c.dedup();
            c
        };
        self.n_classes = classes.len();

        let mut layer_sizes = Vec::new();
        layer_sizes.push(n_features);
        layer_sizes.extend_from_slice(&self.hidden_layer_sizes);
        layer_sizes.push(self.n_classes);

        let mut network = MLPNetwork::new(&layer_sizes, &self.activation);
        let y_onehot = one_hot(y, self.n_classes);

        for _epoch in 0..self.max_iter {
            let batches = minibatch_indices(n_samples, self.batch_size);
            for batch_indices in &batches {
                let batch_size = batch_indices.len();
                let mut x_batch = Array2::zeros((batch_size, n_features));
                let mut y_batch = Array2::zeros((batch_size, self.n_classes));
                for (bi, &idx) in batch_indices.iter().enumerate() {
                    for j in 0..n_features {
                        x_batch[[bi, j]] = x[[idx, j]];
                    }
                    for j in 0..self.n_classes {
                        y_batch[[bi, j]] = y_onehot[[idx, j]];
                    }
                }

                let (pre_acts, acts) = network.forward(&x_batch);
                let n_layers = network.weights.len();

                let mut deltas: Vec<Array2<f64>> = Vec::with_capacity(n_layers);

                let output = softmax(&acts[n_layers - 1]);
                let delta_out = &output - &y_batch;
                deltas.push(delta_out);

                for i in (0..n_layers - 1).rev() {
                    let d = deltas.last().unwrap();
                    let d_hidden = d.dot(&network.weights[i + 1].t()) * &network.activate_grad(&pre_acts[i]);
                    deltas.push(d_hidden);
                }
                deltas.reverse();

                for i in 0..n_layers {
                    let prev_act = if i == 0 {
                        x_batch.clone()
                    } else {
                        acts[i - 1].clone()
                    };
                    let dw = prev_act.t().dot(&deltas[i]) / (batch_size as f64)
                        + &network.weights[i] * self.alpha;
                    let db = deltas[i].sum_axis(ndarray::Axis(0)) / (batch_size as f64);
                    network.weights[i] = &network.weights[i] - &dw * self.learning_rate;
                    network.biases[i] = &network.biases[i] - &db * self.learning_rate;
                }
            }
        }

        self.network = Some(network);
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let network = self.network.as_ref().expect("MLPClassifier not fitted");
        let (_, acts) = network.forward(x);
        let n_layers = acts.len();
        softmax(&acts[n_layers - 1])
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let proba = self.predict_proba(x);
        let n = proba.nrows();
        let mut preds = Array1::zeros(n);
        for i in 0..n {
            let mut best = 0;
            let mut best_val = proba[[i, 0]];
            for j in 1..self.n_classes {
                if proba[[i, j]] > best_val {
                    best_val = proba[[i, j]];
                    best = j;
                }
            }
            preds[i] = best as f64;
        }
        preds
    }
}

pub struct MLPRegressor {
    pub hidden_layer_sizes: Vec<usize>,
    pub activation: String,
    pub learning_rate: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub alpha: f64,
    network: Option<MLPNetwork>,
}

impl MLPRegressor {
    pub fn new(
        hidden_layer_sizes: Vec<usize>,
        activation: &str,
        learning_rate: f64,
        max_iter: usize,
    ) -> Self {
        MLPRegressor {
            hidden_layer_sizes,
            activation: activation.to_string(),
            learning_rate,
            max_iter,
            tol: 1e-4,
            alpha: 0.0001,
            network: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (n_samples, n_features) = x.dim();
        let mut layer_sizes = Vec::new();
        layer_sizes.push(n_features);
        layer_sizes.extend_from_slice(&self.hidden_layer_sizes);
        layer_sizes.push(1);

        let mut network = MLPNetwork::new(&layer_sizes, &self.activation);
        for _epoch in 0..self.max_iter {
            let batch_size = 200.min(n_samples);
            let batches = minibatch_indices(n_samples, batch_size);
            for batch_indices in &batches {
                let bs = batch_indices.len();
                let mut x_batch = Array2::zeros((bs, n_features));
                let mut y_batch = Array2::zeros((bs, 1));
                for (bi, &idx) in batch_indices.iter().enumerate() {
                    for j in 0..n_features {
                        x_batch[[bi, j]] = x[[idx, j]];
                    }
                    y_batch[[bi, 0]] = y[idx];
                }

                let (pre_acts, acts) = network.forward(&x_batch);
                let n_layers = network.weights.len();

                let mut deltas: Vec<Array2<f64>> = Vec::with_capacity(n_layers);

                let delta_out = &acts[n_layers - 1] - &y_batch;
                deltas.push(delta_out);

                for i in (0..n_layers - 1).rev() {
                    let d = deltas.last().unwrap();
                    let d_hidden = d.dot(&network.weights[i + 1].t()) * &network.activate_grad(&pre_acts[i]);
                    deltas.push(d_hidden);
                }
                deltas.reverse();

                for i in 0..n_layers {
                    let prev_act = if i == 0 {
                        x_batch.clone()
                    } else {
                        acts[i - 1].clone()
                    };
                    let dw = prev_act.t().dot(&deltas[i]) / (bs as f64)
                        + &network.weights[i] * self.alpha;
                    let db = deltas[i].sum_axis(ndarray::Axis(0)) / (bs as f64);
                    network.weights[i] = &network.weights[i] - &dw * self.learning_rate;
                    network.biases[i] = &network.biases[i] - &db * self.learning_rate;
                }
            }
        }

        self.network = Some(network);
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let network = self.network.as_ref().expect("MLPRegressor not fitted");
        let (_, acts) = network.forward(x);
        let n_layers = acts.len();
        let output = &acts[n_layers - 1];
        output.column(0).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn make_iris_like_data() -> (Array2<f64>, Array1<f64>) {
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            x_data.push(rng.gen_range(-0.5..0.5));
            x_data.push(rng.gen_range(-0.5..0.5));
            y_data.push(0.0);
        }
        for _ in 0..50 {
            x_data.push(rng.gen_range(2.5..3.5));
            x_data.push(rng.gen_range(-0.5..0.5));
            y_data.push(1.0);
        }
        for _ in 0..50 {
            x_data.push(rng.gen_range(1.0..2.0));
            x_data.push(rng.gen_range(2.5..3.5));
            y_data.push(2.0);
        }
        let x = Array2::from_shape_vec((150, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        (x, y)
    }

    fn make_xor_data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        let y = array![0.0, 1.0, 1.0, 0.0];
        let mut x_aug = Vec::new();
        let mut y_aug = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            for i in 0..4 {
                x_aug.push(x[[i, 0]] + rng.gen_range(-0.05..0.05));
                x_aug.push(x[[i, 1]] + rng.gen_range(-0.05..0.05));
                y_aug.push(y[i]);
            }
        }
        let x_arr = Array2::from_shape_vec((200, 2), x_aug).unwrap();
        let y_arr = Array1::from_vec(y_aug);
        (x_arr, y_arr)
    }

    fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let correct = y_true.iter().zip(y_pred.iter()).filter(|&(a, b)| (*a - *b).abs() < 0.5).count();
        correct as f64 / y_true.len() as f64
    }

    fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let mean = y_true.mean().unwrap();
        let ss_res: f64 = y_true.iter().zip(y_pred.iter()).map(|(a, b)| (*a - *b).powi(2)).sum();
        let ss_tot: f64 = y_true.iter().map(|a| (*a - mean).powi(2)).sum();
        1.0 - ss_res / ss_tot
    }

    #[test]
    fn test_mlp_classifier_iris_relu() {
        let (x, y) = make_iris_like_data();
        let mut clf = MLPClassifier::new(vec![20, 10], "relu", 0.01, 300);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        let acc = accuracy(&y, &preds);
        assert!(acc >= 0.8, "accuracy = {acc}");
    }

    #[test]
    fn test_mlp_classifier_iris_tanh() {
        let (x, y) = make_iris_like_data();
        let mut clf = MLPClassifier::new(vec![20, 10], "tanh", 0.01, 300);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        let acc = accuracy(&y, &preds);
        assert!(acc >= 0.8, "accuracy = {acc}");
    }

    #[test]
    fn test_mlp_classifier_xor() {
        let (x, y) = make_xor_data();
        let mut clf = MLPClassifier::new(vec![8, 8], "relu", 0.01, 1000);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        let acc = accuracy(&y, &preds);
        assert!(acc > 0.7, "accuracy = {acc}");
    }

    #[test]
    fn test_mlp_regressor_squared() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for _ in 0..200 {
            let v = rng.gen_range(-2.0..2.0);
            x_data.push(v);
            y_data.push(v * v + rng.gen_range(-0.1..0.1));
        }
        let x = Array2::from_shape_vec((200, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        let mut reg = MLPRegressor::new(vec![32, 16], "relu", 0.01, 800);
        reg.fit(&x, &y);
        let preds = reg.predict(&x);
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.3, "R² = {r2}");
    }

    #[test]
    fn test_mlp_regressor_tanh() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for _ in 0..200 {
            let v = rng.gen_range(-2.0..2.0);
            x_data.push(v);
            y_data.push(v * v + rng.gen_range(-0.1..0.1));
        }
        let x = Array2::from_shape_vec((200, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        let mut reg = MLPRegressor::new(vec![32, 16], "tanh", 0.01, 1000);
        reg.fit(&x, &y);
        let preds = reg.predict(&x);
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.3, "R² = {r2}");
    }
}
