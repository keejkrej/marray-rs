use ndarray::Array2;
use rand::Rng;

/// t-Distributed Stochastic Neighbor Embedding (t-SNE).
pub struct TSNE {
    pub n_components: usize,
    pub perplexity: f64,
    pub early_exaggeration: f64,
    pub learning_rate: f64,
    pub n_iter: usize,
    pub embedding: Option<Array2<f64>>,
}

impl TSNE {
    pub fn new(
        n_components: usize,
        perplexity: Option<f64>,
        learning_rate: Option<f64>,
        n_iter: Option<usize>,
    ) -> Self {
        Self {
            n_components,
            perplexity: perplexity.unwrap_or(30.0),
            early_exaggeration: 12.0,
            learning_rate: learning_rate.unwrap_or(200.0),
            n_iter: n_iter.unwrap_or(500),
            embedding: None,
        }
    }

    /// Compute high-dimensional affinities P with binary search for sigma to match target perplexity.
    fn compute_p(&self, x: &Array2<f64>) -> Array2<f64> {
        let (n, d) = x.dim();
        let target_entropy = self.perplexity.ln();
        let tol = 1e-4;

        // Pairwise squared Euclidean distances
        let mut d2 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let dist_sq: f64 = (0..d).map(|k| (x[[i, k]] - x[[j, k]]).powi(2)).sum();
                d2[[i, j]] = dist_sq;
            }
        }

        let mut p_cond = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            let mut beta_min = f64::NEG_INFINITY;
            let mut beta_max = f64::INFINITY;
            let mut beta = 1.0; // beta = 1 / (2 * sigma^2)

            for _ in 0..50 {
                let mut sum_pi = 0.0;
                for j in 0..n {
                    if i != j {
                        let val = (-beta * d2[[i, j]]).exp();
                        p_cond[[i, j]] = val;
                        sum_pi += val;
                    } else {
                        p_cond[[i, j]] = 0.0;
                    }
                }

                if sum_pi < 1e-15 {
                    sum_pi = 1e-15;
                }

                let mut h = 0.0;
                for j in 0..n {
                    if i != j {
                        p_cond[[i, j]] /= sum_pi;
                        let p = p_cond[[i, j]];
                        if p > 1e-12 {
                            h -= p * p.ln();
                        }
                    }
                }

                let h_diff = h - target_entropy;
                if h_diff.abs() < tol {
                    break;
                }

                if h_diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() {
                        beta * 2.0
                    } else {
                        (beta + beta_max) * 0.5
                    };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_infinite() {
                        beta * 0.5
                    } else {
                        (beta + beta_min) * 0.5
                    };
                }
            }
        }

        // Symmetrize: P = (P + P^T) / (2 * n)
        let mut p_sym = Array2::<f64>::zeros((n, n));
        let scale = 1.0 / (2.0 * n as f64);
        for i in 0..n {
            for j in 0..n {
                p_sym[[i, j]] = (p_cond[[i, j]] + p_cond[[j, i]]) * scale;
            }
        }

        p_sym
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (n, _) = x.dim();
        if n < 4 {
            return Err("t-SNE requires at least 4 points".to_string());
        }

        let p = self.compute_p(x);
        let mut rng = rand::thread_rng();

        // Initialize embedding Y randomly from N(0, 1e-4)
        let mut y = Array2::from_shape_fn((n, self.n_components), |_| rng.gen_range(-1e-4..1e-4));
        let mut y_m1 = y.clone();
        let mut y_m2 = y.clone();

        let stop_lying_iter = 100.min(self.n_iter / 2);

        for iter in 0..self.n_iter {
            let momentum = if iter < 250 { 0.5 } else { 0.8 };
            let p_mult = if iter < stop_lying_iter {
                self.early_exaggeration
            } else {
                1.0
            };

            // Compute Q matrix
            let mut num = Array2::<f64>::zeros((n, n));
            let mut sum_num = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d_sq: f64 = (0..self.n_components)
                        .map(|c| {
                            let diff = y[[i, c]] - y[[j, c]];
                            diff * diff
                        })
                        .sum();
                    let val = 1.0 / (1.0 + d_sq);
                    num[[i, j]] = val;
                    num[[j, i]] = val;
                    sum_num += 2.0 * val;
                }
            }

            sum_num = sum_num.max(1e-12);
            let mut q = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        q[[i, j]] = (num[[i, j]] / sum_num).max(1e-12);
                    }
                }
            }

            // Compute gradients: dY_i = 4 * sum_j (p_ij * mult - q_ij) * num_ij * (y_i - y_j)
            let mut dy = Array2::<f64>::zeros((n, self.n_components));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let mult = 4.0 * (p[[i, j]] * p_mult - q[[i, j]]) * num[[i, j]];
                        for c in 0..self.n_components {
                            dy[[i, c]] += mult * (y[[i, c]] - y[[j, c]]);
                        }
                    }
                }
            }

            // Update with momentum
            for i in 0..n {
                for c in 0..self.n_components {
                    let update = self.learning_rate * dy[[i, c]]
                        + momentum * (y[[i, c]] - y_m1[[i, c]])
                        + 0.0001 * (y_m1[[i, c]] - y_m2[[i, c]]);
                    y_m2[[i, c]] = y_m1[[i, c]];
                    y_m1[[i, c]] = y[[i, c]];
                    y[[i, c]] -= update;
                }
            }

            // Zero-mean embedding
            let mean = y.mean_axis(ndarray::Axis(0)).unwrap();
            for i in 0..n {
                for c in 0..self.n_components {
                    y[[i, c]] -= mean[c];
                }
            }
        }

        self.embedding = Some(y.clone());
        Ok(y)
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
        self.fit_transform(x)?;
        Ok(())
    }
}

/// Isometric Feature Mapping (Isomap).
pub struct Isomap {
    pub n_neighbors: usize,
    pub n_components: usize,
    pub embedding: Option<Array2<f64>>,
}

impl Isomap {
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Self {
            n_neighbors,
            n_components,
            embedding: None,
        }
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (n, d) = x.dim();
        if n < self.n_neighbors + 1 {
            return Err("n_samples must be > n_neighbors".to_string());
        }

        // 1. Build k-NN distance graph
        let mut d_graph = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n {
            d_graph[i][i] = 0.0;
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = (0..d)
                        .map(|k| (x[[i, k]] - x[[j, k]]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (j, dist)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(neighbor, dist) in dists.iter().take(self.n_neighbors) {
                d_graph[i][neighbor] = dist;
                d_graph[neighbor][i] = dist;
            }
        }

        // 2. All-pairs shortest path (Floyd-Warshall)
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if d_graph[i][k] + d_graph[k][j] < d_graph[i][j] {
                        d_graph[i][j] = d_graph[i][k] + d_graph[k][j];
                    }
                }
            }
        }

        // 3. Classical MDS (Multidimensional Scaling)
        // D2 matrix: D_ij^2
        let mut d2 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let dist = d_graph[i][j];
                d2[[i, j]] = if dist.is_finite() { dist.powi(2) } else { 1e6 };
            }
        }

        // Double centering: B = -0.5 * H * D2 * H where H = I - 1/n * 1 1^T
        let row_means = d2.mean_axis(ndarray::Axis(1)).unwrap();
        let col_means = d2.mean_axis(ndarray::Axis(0)).unwrap();
        let total_mean = d2.mean().unwrap();

        let mut b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = -0.5 * (d2[[i, j]] - row_means[i] - col_means[j] + total_mean);
            }
        }

        // Eigendecomposition / SVD of B
        let svd = crate::sp::linalg::svd(&b);
        let u = svd.u.ok_or_else(|| "SVD failed for MDS".to_string())?;
        let s = svd.s;

        let k = self.n_components.min(n);
        let mut y = Array2::<f64>::zeros((n, k));
        for c in 0..k {
            let scale = s[c].max(0.0).sqrt();
            for i in 0..n {
                y[[i, c]] = u[[i, c]] * scale;
            }
        }

        self.embedding = Some(y.clone());
        Ok(y)
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
        self.fit_transform(x)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_tsne() {
        let x = array![
            [1.0, 1.0, 1.0],
            [1.1, 1.0, 1.2],
            [10.0, 10.0, 10.0],
            [10.1, 10.2, 10.0]
        ];
        let mut tsne = TSNE::new(2, Some(2.0), Some(100.0), Some(50));
        let y = tsne.fit_transform(&x).unwrap();
        assert_eq!(y.dim(), (4, 2));
    }

    #[test]
    fn test_isomap() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];
        let mut iso = Isomap::new(2, 1);
        let y = iso.fit_transform(&x).unwrap();
        assert_eq!(y.dim(), (5, 1));
    }
}
