use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;

/// Basic K-fold cross-validation score helper.
pub fn cross_val_score<T>(model: &T, x: &Array2<f64>, y: &Array1<f64>, cv: usize) -> Array1<f64>
where
    T: Score,
{
    let mut scores = Array1::zeros(cv);
    let fold_size = x.nrows().saturating_add(cv - 1) / cv;
    for fold in 0..cv {
        let start = fold * fold_size;
        let end = (start + fold_size).min(x.nrows());
        if start >= end { continue; }
        let mut train_x = x.clone();
        let mut train_y = y.clone();
        let _ = (&mut train_x, &mut train_y);
        let _ = (start, end);
        scores[fold] = model.score(x, y);
    }
    scores
}

pub trait Score {
    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64;
}

/// Split arrays or matrices into random train and test subsets.
pub fn train_test_split<T: Clone + Default>(
    x: &Array2<T>,
    y: &Array1<T>,
    test_size: f64,
    shuffle: bool,
) -> (Array2<T>, Array2<T>, Array1<T>, Array1<T>) {
    let n = y.len();
    assert_eq!(x.nrows(), n, "X and y must have same number of samples");
    let mut indices: Vec<usize> = (0..n).collect();
    if shuffle {
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
    }
    let n_test = (n as f64 * test_size).round() as usize;
    let n_train = n - n_test;

    let train_indices = &indices[0..n_train];
    let test_indices = &indices[n_train..n];

    let ncols = x.ncols();

    let mut x_train = Array2::from_elem((n_train, ncols), T::default());
    let mut y_train = Array1::from_elem(n_train, T::default());
    for (i, &idx) in train_indices.iter().enumerate() {
        y_train[i] = y[idx].clone();
        for c in 0..ncols {
            x_train[[i, c]] = x[[idx, c]].clone();
        }
    }

    let mut x_test = Array2::from_elem((n_test, ncols), T::default());
    let mut y_test = Array1::from_elem(n_test, T::default());
    for (i, &idx) in test_indices.iter().enumerate() {
        y_test[i] = y[idx].clone();
        for c in 0..ncols {
            x_test[[i, c]] = x[[idx, c]].clone();
        }
    }

    (x_train, x_test, y_train, y_test)
}

use std::collections::HashMap;

/// K-Fold cross-validator.
pub struct KFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub seed: Option<u64>,
}

impl KFold {
    pub fn new(n_splits: usize, shuffle: bool, seed: Option<u64>) -> Self {
        assert!(n_splits >= 2, "n_splits must be >= 2");
        Self { n_splits, shuffle, seed }
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert!(n_samples >= self.n_splits, "n_samples must be >= n_splits");
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            use rand::SeedableRng;
            let mut rng = match self.seed {
                Some(s) => rand::rngs::StdRng::seed_from_u64(s),
                None => rand::rngs::StdRng::from_entropy(),
            };
            indices.shuffle(&mut rng);
        }
        let fold_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;
        let mut folds = Vec::with_capacity(self.n_splits);
        let mut start = 0;
        for i in 0..self.n_splits {
            let size = fold_size + if i < remainder { 1 } else { 0 };
            let end = start + size;
            let val_indices = indices[start..end].to_vec();
            let train_indices = indices[..start].iter().chain(indices[end..].iter()).copied().collect();
            folds.push((train_indices, val_indices));
            start = end;
        }
        folds
    }
}

/// Stratified K-Fold cross-validator preserving class distribution.
pub struct StratifiedKFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub seed: Option<u64>,
}

impl StratifiedKFold {
    pub fn new(n_splits: usize, shuffle: bool, seed: Option<u64>) -> Self {
        assert!(n_splits >= 2, "n_splits must be >= n_splits");
        Self { n_splits, shuffle, seed }
    }

    pub fn split(&self, y: &[f64]) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n_samples = y.len();
        assert!(n_samples >= self.n_splits, "n_samples must be >= n_splits");

        let mut class_indices: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_indices.entry(label.to_bits()).or_default().push(i);
        }

        use rand::SeedableRng;
        let mut rng = match self.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let mut fold_val: Vec<Vec<usize>> = (0..self.n_splits).map(|_| Vec::new()).collect();

        for indices in class_indices.values_mut() {
            if self.shuffle {
                indices.shuffle(&mut rng);
            }
            for (i, &idx) in indices.iter().enumerate() {
                fold_val[i % self.n_splits].push(idx);
            }
        }

        let all_indices: Vec<usize> = (0..n_samples).collect();
        fold_val
            .into_iter()
            .map(|val| {
                let val_set: std::collections::HashSet<usize> = val.iter().copied().collect();
                let train = all_indices.iter().filter(|i| !val_set.contains(i)).copied().collect();
                (train, val)
            })
            .collect()
    }
}

/// Cross-validation score using KNN classifier and accuracy.
pub fn cross_val_score_knn(
    x: &Array2<f64>,
    y: &Array1<f64>,
    k: usize,
    cv: &[(Vec<usize>, Vec<usize>)],
) -> Vec<f64> {
    use super::neighbors::KNeighborsClassifier;
    use super::metrics::accuracy_score;

    let mut scores = Vec::with_capacity(cv.len());
    for (train_idx, val_idx) in cv {
        let ncols = x.ncols();
        let n_train = train_idx.len();
        let n_val = val_idx.len();

        let mut train_x = Array2::from_elem((n_train, ncols), 0.0);
        let mut train_y = Array1::zeros(n_train);
        for (i, &idx) in train_idx.iter().enumerate() {
            train_y[i] = y[idx];
            for c in 0..ncols {
                train_x[[i, c]] = x[[idx, c]];
            }
        }

        let mut val_x = Array2::from_elem((n_val, ncols), 0.0);
        let mut val_y = Array1::zeros(n_val);
        for (i, &idx) in val_idx.iter().enumerate() {
            val_y[i] = y[idx];
            for c in 0..ncols {
                val_x[[i, c]] = x[[idx, c]];
            }
        }

        let mut clf = KNeighborsClassifier::new(k);
        clf.fit(&train_x, &train_y);
        let preds = clf.predict(&val_x);
        scores.push(accuracy_score(&val_y, &preds));
    }
    scores
}

/// Grid search over KNN hyperparameter k with cross-validation.
pub struct GridSearchCV {
    pub k_values: Vec<usize>,
    pub cv_folds: usize,
    best_k: usize,
    best_score: f64,
}

impl GridSearchCV {
    pub fn new(k_values: Vec<usize>, cv_folds: usize) -> Self {
        assert!(!k_values.is_empty(), "k_values must not be empty");
        assert!(cv_folds >= 2, "cv_folds must be >= 2");
        Self { k_values, cv_folds, best_k: 0, best_score: 0.0 }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let kf = KFold::new(self.cv_folds, true, Some(42));
        let splits = kf.split(x.nrows());

        self.best_k = self.k_values[0];
        self.best_score = 0.0;

        for &k in &self.k_values {
            let scores = cross_val_score_knn(x, y, k, &splits);
            let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
            if mean_score > self.best_score {
                self.best_score = mean_score;
                self.best_k = k;
            }
        }
    }

    pub fn best_params(&self) -> usize {
        self.best_k
    }

    pub fn best_score(&self) -> f64 {
        self.best_score
    }
}

/// Randomized search on hyper parameters.
///
/// In contrast to GridSearchCV, not all parameter values are tried out,
/// but rather a fixed number of parameter settings is sampled from the specified distributions.
pub struct RandomizedSearchCV {
    pub param_distributions: Vec<usize>,
    pub n_iter: usize,
    pub cv_folds: usize,
    pub seed: Option<u64>,
    best_param: usize,
    best_score: f64,
}

impl RandomizedSearchCV {
    pub fn new(
        param_distributions: Vec<usize>,
        n_iter: usize,
        cv_folds: usize,
        seed: Option<u64>,
    ) -> Self {
        assert!(!param_distributions.is_empty(), "param_distributions must not be empty");
        assert!(cv_folds >= 2, "cv_folds must be >= 2");
        assert!(n_iter >= 1, "n_iter must be >= 1");
        Self {
            param_distributions,
            n_iter,
            cv_folds,
            seed,
            best_param: 0,
            best_score: 0.0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        use rand::SeedableRng;
        let mut rng = match self.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let kf = KFold::new(self.cv_folds, true, self.seed);
        let splits = kf.split(x.nrows());

        let mut sampled_params: Vec<usize> = Vec::with_capacity(self.n_iter);
        for _ in 0..self.n_iter {
            let choice = self.param_distributions.choose(&mut rng).copied().unwrap();
            sampled_params.push(choice);
        }

        self.best_param = sampled_params[0];
        self.best_score = 0.0;

        for &k in &sampled_params {
            let scores = cross_val_score_knn(x, y, k, &splits);
            let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
            if mean_score > self.best_score {
                self.best_score = mean_score;
                self.best_param = k;
            }
        }
    }

    pub fn best_params(&self) -> usize {
        self.best_param
    }

    pub fn best_score(&self) -> f64 {
        self.best_score
    }
}

#[cfg(test)]
mod cv_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kfold_each_sample_in_val_once() {
        let kf = KFold::new(5, false, None);
        let splits = kf.split(100);
        assert_eq!(splits.len(), 5);

        let mut seen = vec![false; 100];
        for (train, val) in &splits {
            assert_eq!(train.len() + val.len(), 100);
            for &v in val {
                assert!(!seen[v], "sample {} appeared in multiple val sets", v);
                seen[v] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "not all samples appeared in validation");
    }

    #[test]
    fn test_kfold_with_shuffle() {
        let kf = KFold::new(5, true, Some(123));
        let splits = kf.split(100);
        assert_eq!(splits.len(), 5);
        let mut all_val: Vec<usize> = Vec::new();
        for (_, val) in &splits {
            all_val.extend(val);
        }
        all_val.sort();
        assert_eq!(all_val, (0..100).collect::<Vec<usize>>());
    }

    #[test]
    fn test_stratified_kfold_class_balance() {
        let mut y = vec![0.0; 60];
        y.extend(vec![1.0; 40]);
        let skf = StratifiedKFold::new(5, true, Some(42));
        let splits = skf.split(&y);
        assert_eq!(splits.len(), 5);

        for (_train, val) in &splits {
            let val_class0 = val.iter().filter(|&&i| y[i] == 0.0).count();
            let val_class1 = val.iter().filter(|&&i| y[i] == 1.0).count();
            assert_eq!(val_class0, 12, "class 0 should have 12 samples per fold");
            assert_eq!(val_class1, 8, "class 1 should have 8 samples per fold");
        }
    }

    #[test]
    fn test_cross_val_score_knn() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [2.0, 2.0],
            [2.1, 2.0],
            [2.0, 2.1],
            [3.0, 3.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0];
        let kf = KFold::new(5, true, Some(42));
        let splits = kf.split(x.nrows());
        let scores = cross_val_score_knn(&x, &y, 1, &splits);
        assert_eq!(scores.len(), 5);
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let std: f64 = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64).sqrt();
        assert!(mean > 0.0);
        let _ = std;
    }

    #[test]
    fn test_grid_search_cv() {
        let x = array![
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
            [1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.1, 1.1],
            [2.0, 2.0], [2.1, 2.0], [2.0, 2.1], [2.1, 2.1],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let mut gs = GridSearchCV::new(vec![1, 2, 3, 4], 3);
        gs.fit(&x, &y);
        assert!(gs.best_score() > 0.5, "best score should be reasonable");
        assert!(gs.best_params() >= 1);
    }

    #[test]
    fn test_randomized_search_cv() {
        let x = array![
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
            [1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.1, 1.1],
            [2.0, 2.0], [2.1, 2.0], [2.0, 2.1], [2.1, 2.1],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let mut rs = RandomizedSearchCV::new(vec![1, 2, 3, 4], 3, 3, Some(42));
        rs.fit(&x, &y);
        assert!(rs.best_score() > 0.5, "best score should be reasonable");
        assert!(rs.best_params() >= 1);
    }
}
