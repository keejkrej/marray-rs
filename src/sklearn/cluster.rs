use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;

/// DBSCAN density-based clustering.
pub struct DBSCAN {
    pub eps: f64,
    pub min_samples: usize,
}

impl DBSCAN {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    pub fn fit_predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let nrows = x.nrows();
        let ncols = x.ncols();
        let eps_sq = self.eps * self.eps;

        let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let mut nbrs = Vec::new();
            for j in 0..nrows {
                if i == j {
                    continue;
                }
                let mut dist_sq = 0.0;
                for c in 0..ncols {
                    dist_sq += (x[[i, c]] - x[[j, c]]).powi(2);
                }
                if dist_sq <= eps_sq {
                    nbrs.push(j);
                }
            }
            neighbors.push(nbrs);
        }

        let mut labels = Array1::from_elem(nrows, -1.0f64);
        let mut visited = vec![false; nrows];
        let mut cluster_id = 0.0f64;

        for i in 0..nrows {
            if visited[i] {
                continue;
            }
            visited[i] = true;

            if neighbors[i].len() < self.min_samples {
                continue;
            }

            labels[i] = cluster_id;
            let mut seed_set: Vec<usize> = neighbors[i].clone();
            let mut idx = 0;
            while idx < seed_set.len() {
                let q = seed_set[idx];
                if !visited[q] {
                    visited[q] = true;
                    if neighbors[q].len() >= self.min_samples {
                        for &nbr in &neighbors[q] {
                            if !seed_set.contains(&nbr) {
                                seed_set.push(nbr);
                            }
                        }
                    }
                }
                if labels[q] < 0.0 {
                    labels[q] = cluster_id;
                }
                idx += 1;
            }

            cluster_id += 1.0;
        }

        labels
    }
}

/// K-Means clustering.
pub struct KMeans {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub cluster_centers: Option<Array2<f64>>,
}

impl KMeans {
    pub fn new(n_clusters: usize, max_iter: usize) -> Self {
        Self {
            n_clusters,
            max_iter,
            tol: 1e-4,
            cluster_centers: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let (nrows, ncols) = x.dim();
        if nrows < self.n_clusters {
            panic!("Number of samples must be >= n_clusters");
        }

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..nrows).collect();
        indices.shuffle(&mut rng);

        let mut centroids = Array2::zeros((self.n_clusters, ncols));
        for k in 0..self.n_clusters {
            let idx = indices[k];
            for c in 0..ncols {
                centroids[[k, c]] = x[[idx, c]];
            }
        }

        let mut labels = vec![0; nrows];

        for _ in 0..self.max_iter {
            let mut changed = false;
            for r in 0..nrows {
                let mut min_dist = f64::INFINITY;
                let mut best_k = 0;
                for k in 0..self.n_clusters {
                    let mut dist_sq = 0.0;
                    for c in 0..ncols {
                        dist_sq += (x[[r, c]] - centroids[[k, c]]).powi(2);
                    }
                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                        best_k = k;
                    }
                }
                if labels[r] != best_k {
                    labels[r] = best_k;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            let mut new_centroids: Array2<f64> = Array2::zeros((self.n_clusters, ncols));
            let mut counts = vec![0.0; self.n_clusters];
            for r in 0..nrows {
                let k = labels[r];
                counts[k] += 1.0;
                for c in 0..ncols {
                    new_centroids[[k, c]] += x[[r, c]];
                }
            }

            let mut centroid_diff: f64 = 0.0;
            for k in 0..self.n_clusters {
                if counts[k] > 0.0 {
                    for c in 0..ncols {
                        let new_val = new_centroids[[k, c]] / counts[k];
                        let diff = centroids[[k, c]] - new_val;
                        centroid_diff += diff * diff;
                        centroids[[k, c]] = new_val;
                    }
                }
            }

            if centroid_diff.sqrt() < self.tol {
                break;
            }
        }

        self.cluster_centers = Some(centroids);
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let centroids = self.cluster_centers.as_ref().expect("KMeans model not fitted");
        let (nrows, ncols) = x.dim();
        let mut labels = Array1::zeros(nrows);
        for r in 0..nrows {
            let mut min_dist = f64::INFINITY;
            let mut best_k = 0;
            for k in 0..self.n_clusters {
                let mut dist_sq = 0.0;
                for c in 0..ncols {
                    dist_sq += (x[[r, c]] - centroids[[k, c]]).powi(2);
                }
                if dist_sq < min_dist {
                    min_dist = dist_sq;
                    best_k = k;
                }
            }
            labels[r] = best_k;
        }
        labels
    }
}

/// Agglomerative Hierarchical Clustering.
///
/// Recursively merges pairs of clusters based on a linkage criterion.
pub struct AgglomerativeClustering {
    pub n_clusters: usize,
    pub linkage: String,
}

impl AgglomerativeClustering {
    pub fn new(n_clusters: usize, linkage: Option<&str>) -> Self {
        Self {
            n_clusters: n_clusters.max(1),
            linkage: linkage.unwrap_or("ward").to_lowercase(),
        }
    }

    pub fn fit_predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Array1::zeros(0);
        }
        if n_samples <= self.n_clusters {
            return Array1::from_iter(0..n_samples);
        }

        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();

        let mut pairwise_dist = Array2::<f64>::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let mut d_sq = 0.0;
                for c in 0..x.ncols() {
                    let diff = x[[i, c]] - x[[j, c]];
                    d_sq += diff * diff;
                }
                let d = d_sq.sqrt();
                pairwise_dist[[i, j]] = d;
                pairwise_dist[[j, i]] = d;
            }
        }

        let cluster_dist = |c1: &[usize], c2: &[usize], linkage: &str| -> f64 {
            match linkage {
                "single" => {
                    let mut min_d = f64::INFINITY;
                    for &i in c1 {
                        for &j in c2 {
                            min_d = min_d.min(pairwise_dist[[i, j]]);
                        }
                    }
                    min_d
                }
                "complete" => {
                    let mut max_d = 0.0f64;
                    for &i in c1 {
                        for &j in c2 {
                            max_d = max_d.max(pairwise_dist[[i, j]]);
                        }
                    }
                    max_d
                }
                "average" => {
                    let mut sum_d = 0.0;
                    for &i in c1 {
                        for &j in c2 {
                            sum_d += pairwise_dist[[i, j]];
                        }
                    }
                    sum_d / ((c1.len() * c2.len()) as f64)
                }
                _ => {
                    // Ward: delta_SS = (|c1| * |c2|) / (|c1| + |c2|) * ||mean1 - mean2||^2
                    let mut mean1 = vec![0.0; x.ncols()];
                    for &i in c1 {
                        for c in 0..x.ncols() {
                            mean1[c] += x[[i, c]];
                        }
                    }
                    for c in 0..x.ncols() {
                        mean1[c] /= c1.len() as f64;
                    }

                    let mut mean2 = vec![0.0; x.ncols()];
                    for &i in c2 {
                        for c in 0..x.ncols() {
                            mean2[c] += x[[i, c]];
                        }
                    }
                    for c in 0..x.ncols() {
                        mean2[c] /= c2.len() as f64;
                    }

                    let mut mean_diff_sq = 0.0;
                    for c in 0..x.ncols() {
                        mean_diff_sq += (mean1[c] - mean2[c]).powi(2);
                    }

                    ((c1.len() * c2.len()) as f64 / ((c1.len() + c2.len()) as f64)) * mean_diff_sq
                }
            }
        };

        while clusters.len() > self.n_clusters {
            let mut best_i = 0;
            let mut best_j = 1;
            let mut min_val = f64::INFINITY;

            for i in 0..clusters.len() {
                for j in i + 1..clusters.len() {
                    let d = cluster_dist(&clusters[i], &clusters[j], &self.linkage);
                    if d < min_val {
                        min_val = d;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            let mut merged = clusters[best_i].clone();
            merged.extend_from_slice(&clusters[best_j]);
            clusters.remove(best_j);
            clusters.remove(best_i);
            clusters.push(merged);
        }

        let mut labels = Array1::zeros(n_samples);
        for (cluster_id, cl) in clusters.iter().enumerate() {
            for &sample_idx in cl {
                labels[sample_idx] = cluster_id;
            }
        }

        labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_agglomerative_clustering() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.2],
            [10.0, 10.0],
            [10.1, 10.2],
            [10.2, 10.0]
        ];

        let agg = AgglomerativeClustering::new(2, Some("ward"));
        let labels = agg.fit_predict(&x);

        assert_eq!(labels.len(), 6);
        // Points in cluster 1 should have same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        // Points in cluster 2 should have same label
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        // Different clusters
        assert_ne!(labels[0], labels[3]);
    }
}
