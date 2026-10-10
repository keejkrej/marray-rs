use ndarray::{Array1, Array2};

/// Simple K-nearest neighbors classifier using Euclidean distance.
pub struct KNeighborsClassifier {
    pub n_neighbors: usize,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
}

impl KNeighborsClassifier {
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors, x_train: None, y_train: None }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x_train = self.x_train.as_ref().expect("KNeighborsClassifier not fitted");
        let y_train = self.y_train.as_ref().expect("KNeighborsClassifier not fitted");
        let mut preds = Array1::zeros(x.nrows());
        for r in 0..x.nrows() {
            let mut distances = Vec::new();
            for i in 0..x_train.nrows() {
                let mut dist = 0.0;
                for c in 0..x_train.ncols() {
                    let diff = x[[r, c]] - x_train[[i, c]];
                    dist += diff * diff;
                }
                distances.push((dist.sqrt(), y_train[i]));
            }
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut counts = std::collections::HashMap::new();
            for (dist, label) in distances.into_iter().take(self.n_neighbors) {
                let _ = dist;
                *counts.entry(label.to_bits()).or_insert(0usize) += 1;
            }
            let majority = counts.into_iter().max_by_key(|(_, count)| *count).map(|(bits, _)| f64::from_bits(bits)).unwrap_or(0.0);
            preds[r] = majority;
        }
        preds
    }
}

/// K-nearest neighbors regression.
///
/// Target is predicted by local interpolation of the targets associated with the nearest neighbors.
pub struct KNeighborsRegressor {
    pub n_neighbors: usize,
    pub weights: String,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
}

impl KNeighborsRegressor {
    pub fn new(n_neighbors: usize, weights: Option<&str>) -> Self {
        Self {
            n_neighbors,
            weights: weights.unwrap_or("uniform").to_lowercase(),
            x_train: None,
            y_train: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x_train = self.x_train.as_ref().expect("KNeighborsRegressor not fitted");
        let y_train = self.y_train.as_ref().expect("KNeighborsRegressor not fitted");
        let mut preds = Array1::zeros(x.nrows());

        for r in 0..x.nrows() {
            let mut distances = Vec::new();
            for i in 0..x_train.nrows() {
                let mut dist_sq = 0.0;
                for c in 0..x_train.ncols() {
                    let diff = x[[r, c]] - x_train[[i, c]];
                    dist_sq += diff * diff;
                }
                distances.push((dist_sq.sqrt(), y_train[i]));
            }
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let k_nearest: Vec<(f64, f64)> = distances.into_iter().take(self.n_neighbors).collect();

            if self.weights == "distance" {
                if let Some(&(_, y_val)) = k_nearest.iter().find(|(d, _)| *d < 1e-12) {
                    preds[r] = y_val;
                    continue;
                }
                let mut sum_weights = 0.0;
                let mut sum_weighted_y = 0.0;
                for (d, y_val) in k_nearest {
                    let w = 1.0 / d;
                    sum_weights += w;
                    sum_weighted_y += w * y_val;
                }
                preds[r] = sum_weighted_y / sum_weights;
            } else {
                let sum_y: f64 = k_nearest.iter().map(|(_, y_val)| *y_val).sum();
                preds[r] = sum_y / k_nearest.len() as f64;
            }
        }

        preds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kneighbors_classifier() {
        let mut clf = KNeighborsClassifier::new(1);
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]);
        clf.fit(&x, &y);
        let preds = clf.predict(&Array2::from_shape_vec((1, 2), vec![0.1, 0.1]).unwrap());
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0], 0.0);
    }

    #[test]
    fn test_kneighbors_regressor() {
        let mut reg = KNeighborsRegressor::new(2, Some("uniform"));
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        reg.fit(&x, &y);
        let preds = reg.predict(&Array2::from_shape_vec((1, 1), vec![1.5]).unwrap());
        assert_eq!(preds.len(), 1);
        assert!((preds[0] - 15.0).abs() < 1e-6);

        let mut reg_dist = KNeighborsRegressor::new(2, Some("distance"));
        reg_dist.fit(&x, &y);
        let preds_dist = reg_dist.predict(&Array2::from_shape_vec((1, 1), vec![1.0]).unwrap());
        assert!((preds_dist[0] - 10.0).abs() < 1e-6);
    }
}
