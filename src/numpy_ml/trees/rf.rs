use ndarray::{Array1, Array2};
use rand::Rng;
use super::dt::DecisionTree;

/// Random Forest ensemble of decision trees.
pub struct RandomForest {
    pub n_trees: usize,
    pub max_depth: Option<usize>,
    pub n_feats: Option<usize>,
    pub classifier: bool,
    pub criterion: String,
    pub seed: Option<u64>,
    pub trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn new(
        n_trees: usize,
        max_depth: Option<usize>,
        n_feats: Option<usize>,
        classifier: bool,
        criterion: Option<&str>,
        seed: Option<u64>,
    ) -> Self {
        let criterion = match criterion {
            Some(c) => c.to_lowercase(),
            None => {
                if classifier {
                    "entropy".to_string()
                } else {
                    "mse".to_string()
                }
            }
        };

        Self {
            n_trees,
            max_depth,
            n_feats,
            classifier,
            criterion,
            seed,
            trees: Vec::with_capacity(n_trees),
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (nrows, ncols) = x.dim();
        assert_eq!(nrows, y.len(), "X and y row count must match");

        let mut rng = rand::thread_rng();
        self.trees.clear();

        for i in 0..self.n_trees {
            let tree_seed = self.seed.map(|s| s.wrapping_add(i as u64));
            let mut tree = DecisionTree::new(
                self.classifier,
                self.max_depth,
                self.n_feats,
                Some(&self.criterion),
                tree_seed,
            );

            // Bootstrap sample
            let mut x_samp = Array2::<f64>::zeros((nrows, ncols));
            let mut y_samp = Array1::<f64>::zeros(nrows);
            for r in 0..nrows {
                let idx = rng.r#gen_range(0..nrows);
                for c in 0..ncols {
                    x_samp[[r, c]] = x[[idx, c]];
                }
                y_samp[r] = y[idx];
            }

            tree.fit(&x_samp, &y_samp);
            self.trees.push(tree);
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        assert!(!self.trees.is_empty(), "RandomForest not fitted");
        let n_samples = x.nrows();
        let mut all_preds = Vec::with_capacity(self.trees.len());
        for tree in &self.trees {
            all_preds.push(tree.predict(x));
        }

        let mut out = Array1::<f64>::zeros(n_samples);
        if self.classifier {
            for i in 0..n_samples {
                let mut counts = std::collections::HashMap::new();
                for tree_pred in &all_preds {
                    let cls = tree_pred[i] as usize;
                    *counts.entry(cls).or_insert(0) += 1;
                }
                let best = counts
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(cls, _)| cls as f64)
                    .unwrap_or(0.0);
                out[i] = best;
            }
        } else {
            for i in 0..n_samples {
                let sum: f64 = all_preds.iter().map(|pred| pred[i]).sum();
                out[i] = sum / self.trees.len() as f64;
            }
        }
        out
    }
}
