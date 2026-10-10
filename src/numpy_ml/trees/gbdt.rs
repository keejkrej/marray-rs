use ndarray::{Array1, Array2};
use super::dt::DecisionTree;
use super::losses::{BaseEstimator, CrossEntropyLoss, MSELoss, TreeLoss};

/// Gradient Boosted Decision Tree ensemble.
pub struct GradientBoostedDecisionTree {
    pub n_iter: usize,
    pub max_depth: Option<usize>,
    pub classifier: bool,
    pub learning_rate: f64,
    pub loss: String,
    pub out_dims: usize,
    base_estimators: Vec<Box<dyn BaseEstimator>>,
    trees: Vec<Vec<DecisionTree>>, // [iteration, dimension]
}

impl GradientBoostedDecisionTree {
    pub fn new(
        n_iter: usize,
        max_depth: Option<usize>,
        classifier: bool,
        learning_rate: f64,
        loss: Option<&str>,
    ) -> Self {
        let loss = match loss {
            Some(l) => l.to_lowercase(),
            None => {
                if classifier {
                    "crossentropy".to_string()
                } else {
                    "mse".to_string()
                }
            }
        };

        Self {
            n_iter,
            max_depth,
            classifier,
            learning_rate,
            loss,
            out_dims: 0,
            base_estimators: Vec::new(),
            trees: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (nrows, _) = x.dim();
        assert_eq!(nrows, y.len(), "X and y row count must match");

        let loss_fn: Box<dyn TreeLoss> = if self.loss == "mse" {
            Box::new(MSELoss::new())
        } else {
            Box::new(CrossEntropyLoss::new())
        };

        let y_matrix = if self.classifier {
            let max_cls = y.iter().map(|&v| v as usize).max().unwrap_or(0);
            let n_classes = (max_cls + 1).max(2);
            let mut one_hot = Array2::<f64>::zeros((nrows, n_classes));
            for r in 0..nrows {
                let c = y[r] as usize;
                if c < n_classes {
                    one_hot[[r, c]] = 1.0;
                }
            }
            one_hot
        } else {
            y.clone().into_shape_with_order((nrows, 1)).unwrap()
        };

        let out_dims = y_matrix.ncols();
        self.out_dims = out_dims;

        self.base_estimators.clear();
        self.trees.clear();

        // Fit base estimators
        let mut y_pred = Array2::<f64>::zeros((nrows, out_dims));
        for k in 0..out_dims {
            let col = y_matrix.column(k).to_owned();
            let mut base = loss_fn.base_estimator();
            base.fit(&col);
            let pred = base.predict(nrows);
            for r in 0..nrows {
                y_pred[[r, k]] += pred[r];
            }
            self.base_estimators.push(base);
        }

        // Iteratively fit weak learners on negative gradients
        for _ in 1..self.n_iter {
            let mut iter_trees = Vec::with_capacity(out_dims);
            for k in 0..out_dims {
                let target = y_matrix.column(k).to_owned();
                let pred = y_pred.column(k).to_owned();
                let neg_grad = &loss_fn.grad(&target, &pred) * -1.0;

                let mut tree = DecisionTree::new(
                    false,
                    self.max_depth,
                    None,
                    Some("mse"),
                    None,
                );
                tree.fit(x, &neg_grad);

                let h_pred = tree.predict(x);
                for r in 0..nrows {
                    y_pred[[r, k]] += self.learning_rate * h_pred[r];
                }
                iter_trees.push(tree);
            }
            self.trees.push(iter_trees);
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let nrows = x.nrows();
        let mut y_pred = Array2::<f64>::zeros((nrows, self.out_dims));

        // Base predictions
        for k in 0..self.out_dims {
            let base_pred = self.base_estimators[k].predict(nrows);
            for r in 0..nrows {
                y_pred[[r, k]] += base_pred[r];
            }
        }

        // Tree predictions
        for iter_trees in &self.trees {
            for k in 0..self.out_dims {
                let h_pred = iter_trees[k].predict(x);
                for r in 0..nrows {
                    y_pred[[r, k]] += self.learning_rate * h_pred[r];
                }
            }
        }

        if self.classifier {
            let mut preds = Array1::<f64>::zeros(nrows);
            for r in 0..nrows {
                let mut best_cls = 0;
                let mut best_val = y_pred[[r, 0]];
                for k in 1..self.out_dims {
                    if y_pred[[r, k]] > best_val {
                        best_val = y_pred[[r, k]];
                        best_cls = k;
                    }
                }
                preds[r] = best_cls as f64;
            }
            preds
        } else {
            y_pred.column(0).to_owned()
        }
    }
}
