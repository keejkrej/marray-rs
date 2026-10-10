use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub enum TreeNode {
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    Leaf {
        // For classification: probability distribution over classes
        // For regression: scalar mean value (stored as single-element vec)
        value: Vec<f64>,
    },
}

/// A decision tree model for regression and classification problems (CART).
pub struct DecisionTree {
    pub classifier: bool,
    pub max_depth: Option<usize>,
    pub n_feats: Option<usize>,
    pub criterion: String,
    pub seed: Option<u64>,
    pub n_classes: usize,
    pub root: Option<Box<TreeNode>>,
}

impl DecisionTree {
    pub fn new(
        classifier: bool,
        max_depth: Option<usize>,
        n_feats: Option<usize>,
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
            classifier,
            max_depth,
            n_feats,
            criterion,
            seed,
            n_classes: 0,
            root: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (nrows, ncols) = x.dim();
        assert_eq!(nrows, y.len(), "X and y row count must match");

        if self.classifier {
            self.n_classes = y.iter().map(|&v| v as usize).max().map(|m| m + 1).unwrap_or(0);
        } else {
            self.n_classes = 1;
        }

        let n_feats = self.n_feats.unwrap_or(ncols).min(ncols);
        let mut rng = match self.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::seed_from_u64(rand::random()),
        };

        let indices: Vec<usize> = (0..nrows).collect();
        self.root = Some(Box::new(self.grow(x, y, &indices, 0, n_feats, &mut rng)));
    }

    fn grow(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        depth: usize,
        n_feats: usize,
        rng: &mut rand::rngs::StdRng,
    ) -> TreeNode {
        let n = indices.len();
        if n == 0 {
            return TreeNode::Leaf {
                value: if self.classifier {
                    vec![0.0; self.n_classes]
                } else {
                    vec![0.0]
                },
            };
        }

        let is_pure = if self.classifier {
            let first = y[indices[0]] as usize;
            indices.iter().all(|&i| y[i] as usize == first)
        } else {
            let first = y[indices[0]];
            indices.iter().all(|&i| (y[i] - first).abs() < 1e-12)
        };

        let reached_max_depth = self.max_depth.map(|md| depth >= md).unwrap_or(false);

        if is_pure || reached_max_depth || n <= 1 {
            return self.create_leaf(y, indices);
        }

        let total_feats = x.ncols();
        let mut feat_candidates: Vec<usize> = (0..total_feats).collect();
        if n_feats < total_feats {
            feat_candidates.shuffle(rng);
            feat_candidates.truncate(n_feats);
        }

        if let Some((feat, thresh, left_indices, right_indices)) =
            self.find_best_split(x, y, indices, &feat_candidates)
        {
            if left_indices.is_empty() || right_indices.is_empty() {
                return self.create_leaf(y, indices);
            }

            let left = self.grow(x, y, &left_indices, depth + 1, n_feats, rng);
            let right = self.grow(x, y, &right_indices, depth + 1, n_feats, rng);

            TreeNode::Internal {
                feature: feat,
                threshold: thresh,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            self.create_leaf(y, indices)
        }
    }

    fn create_leaf(&self, y: &Array1<f64>, indices: &[usize]) -> TreeNode {
        let n = indices.len() as f64;
        if self.classifier {
            let mut counts = vec![0.0; self.n_classes];
            for &i in indices {
                let c = y[i] as usize;
                if c < self.n_classes {
                    counts[c] += 1.0;
                }
            }
            let probs = counts.into_iter().map(|cnt| cnt / n.max(1.0)).collect();
            TreeNode::Leaf { value: probs }
        } else {
            let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n.max(1.0);
            TreeNode::Leaf { value: vec![mean] }
        }
    }

    fn impurity(&self, y: &Array1<f64>, indices: &[usize]) -> f64 {
        let n = indices.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }

        match self.criterion.as_str() {
            "gini" => {
                let mut counts = vec![0.0; self.n_classes];
                for &i in indices {
                    let c = y[i] as usize;
                    if c < self.n_classes {
                        counts[c] += 1.0;
                    }
                }
                1.0 - counts.iter().map(|&cnt| (cnt / n).powi(2)).sum::<f64>()
            }
            "entropy" => {
                let mut counts = vec![0.0; self.n_classes];
                for &i in indices {
                    let c = y[i] as usize;
                    if c < self.n_classes {
                        counts[c] += 1.0;
                    }
                }
                -counts
                    .iter()
                    .map(|&cnt| {
                        let p = cnt / n;
                        if p > 0.0 {
                            p * p.log2()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
            }
            "mse" | _ => {
                let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n;
                indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / n
            }
        }
    }

    fn find_best_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        feat_candidates: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let parent_impurity = self.impurity(y, indices);
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        for &feat in feat_candidates {
            let mut vals: Vec<f64> = indices.iter().map(|&i| x[[i, feat]]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();

            if vals.len() <= 1 {
                continue;
            }

            for window in vals.windows(2) {
                let thresh = 0.5 * (window[0] + window[1]);

                let mut left = Vec::new();
                let mut right = Vec::new();
                for &i in indices {
                    if x[[i, feat]] <= thresh {
                        left.push(i);
                    } else {
                        right.push(i);
                    }
                }

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let left_imp = self.impurity(y, &left);
                let right_imp = self.impurity(y, &right);
                let total = indices.len() as f64;
                let child_imp = (left.len() as f64 / total) * left_imp
                    + (right.len() as f64 / total) * right_imp;
                let gain = parent_impurity - child_imp;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feat, thresh, left, right));
                }
            }
        }

        best_split
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let root = self.root.as_ref().expect("DecisionTree not fitted");
        let n = x.nrows();
        let mut preds = Array1::zeros(n);
        for i in 0..n {
            let row = x.row(i);
            preds[i] = self.traverse_predict(row.as_slice().unwrap(), root);
        }
        preds
    }

    pub fn predict_class_probs(&self, x: &Array2<f64>) -> Array2<f64> {
        assert!(self.classifier, "predict_class_probs only defined for classifier");
        let root = self.root.as_ref().expect("DecisionTree not fitted");
        let n = x.nrows();
        let mut probs = Array2::zeros((n, self.n_classes));
        for i in 0..n {
            let row = x.row(i);
            let p = self.traverse_probs(row.as_slice().unwrap(), root);
            for c in 0..self.n_classes {
                probs[[i, c]] = p.get(c).copied().unwrap_or(0.0);
            }
        }
        probs
    }

    fn traverse_predict(&self, x: &[f64], node: &TreeNode) -> f64 {
        match node {
            TreeNode::Leaf { value } => {
                if self.classifier {
                    value
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx as f64)
                        .unwrap_or(0.0)
                } else {
                    value[0]
                }
            }
            TreeNode::Internal { feature, threshold, left, right } => {
                if x[*feature] <= *threshold {
                    self.traverse_predict(x, left)
                } else {
                    self.traverse_predict(x, right)
                }
            }
        }
    }

    fn traverse_probs<'a>(&self, x: &[f64], node: &'a TreeNode) -> &'a [f64] {
        match node {
            TreeNode::Leaf { value } => value.as_slice(),
            TreeNode::Internal { feature, threshold, left, right } => {
                if x[*feature] <= *threshold {
                    self.traverse_probs(x, left)
                } else {
                    self.traverse_probs(x, right)
                }
            }
        }
    }
}
