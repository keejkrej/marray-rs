use ndarray::{Array1, Array2};
use std::cmp::Ordering;

// --- Distance functions ---

/// Euclidean distance between two vectors: $\sqrt{\sum (u_i - v_i)^2}$.
pub fn euclidean(u: &[f64], v: &[f64]) -> f64 {
    assert_eq!(u.len(), v.len(), "Input vectors must have identical lengths");
    u.iter()
        .zip(v.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Manhattan / City block distance: $\sum |u_i - v_i|$.
pub fn manhattan(u: &[f64], v: &[f64]) -> f64 {
    assert_eq!(u.len(), v.len(), "Input vectors must have identical lengths");
    u.iter().zip(v.iter()).map(|(&a, &b)| (a - b).abs()).sum()
}

/// Alias for `manhattan`.
pub fn cityblock(u: &[f64], v: &[f64]) -> f64 {
    manhattan(u, v)
}

/// Cosine distance: $1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$.
pub fn cosine(u: &[f64], v: &[f64]) -> f64 {
    assert_eq!(u.len(), v.len(), "Input vectors must have identical lengths");
    let mut dot = 0.0;
    let mut norm_u = 0.0;
    let mut norm_v = 0.0;
    for (&a, &b) in u.iter().zip(v.iter()) {
        dot += a * b;
        norm_u += a * a;
        norm_v += b * b;
    }
    let denom = (norm_u * norm_v).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (1.0 - dot / denom).max(0.0)
    }
}

/// Chebyshev distance: $\max_i |u_i - v_i|$.
pub fn chebyshev(u: &[f64], v: &[f64]) -> f64 {
    assert_eq!(u.len(), v.len(), "Input vectors must have identical lengths");
    u.iter()
        .zip(v.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// Minkowski distance: $(\sum |u_i - v_i|^p)^{1/p}$.
pub fn minkowski(u: &[f64], v: &[f64], p: f64) -> f64 {
    assert_eq!(u.len(), v.len(), "Input vectors must have identical lengths");
    if p.is_infinite() {
        return chebyshev(u, v);
    }
    assert!(p >= 1.0, "p must be >= 1.0");
    let sum: f64 = u
        .iter()
        .zip(v.iter())
        .map(|(&a, &b)| (a - b).abs().powf(p))
        .sum();
    sum.powf(1.0 / p)
}

// --- Pairwise Distances and Squareform ---

/// Pairwise distances between observations in condensed format.
///
/// Returns a 1D array of length $n(n-1)/2$ representing the upper triangle of the distance matrix.
pub fn pdist(x: &Array2<f64>, metric: &str) -> Array1<f64> {
    let n = x.nrows();
    let m = n * (n - 1) / 2;
    let mut out = Vec::with_capacity(m);

    for i in 0..n {
        let row_i = x.row(i).to_slice().unwrap().to_vec();
        for j in (i + 1)..n {
            let row_j = x.row(j).to_slice().unwrap();
            let d = match metric {
                "manhattan" | "cityblock" => manhattan(&row_i, row_j),
                "cosine" => cosine(&row_i, row_j),
                "chebyshev" => chebyshev(&row_i, row_j),
                _ => euclidean(&row_i, row_j),
            };
            out.push(d);
        }
    }

    Array1::from_vec(out)
}

/// Convert a condensed distance array to a square matrix, or a square matrix to condensed format.
pub enum SquareformInput<'a> {
    Condensed(&'a Array1<f64>),
    Square(&'a Array2<f64>),
}

pub enum SquareformOutput {
    Square(Array2<f64>),
    Condensed(Array1<f64>),
}

pub fn squareform(input: SquareformInput) -> SquareformOutput {
    match input {
        SquareformInput::Condensed(condensed) => {
            let m = condensed.len();
            // m = n*(n-1)/2  =>  n^2 - n - 2m = 0  =>  n = (1 + sqrt(1 + 8m)) / 2
            let n = ((1.0 + (1.0 + 8.0 * m as f64).sqrt()) / 2.0).round() as usize;
            assert_eq!(n * (n - 1) / 2, m, "Invalid condensed array length");

            let mut mat = Array2::<f64>::zeros((n, n));
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let val = condensed[idx];
                    mat[[i, j]] = val;
                    mat[[j, i]] = val;
                    idx += 1;
                }
            }
            SquareformOutput::Square(mat)
        }
        SquareformInput::Square(mat) => {
            let (rows, cols) = mat.dim();
            assert_eq!(rows, cols, "Matrix must be square");
            let m = rows * (rows - 1) / 2;
            let mut out = Vec::with_capacity(m);
            for i in 0..rows {
                for j in (i + 1)..cols {
                    out.push(mat[[i, j]]);
                }
            }
            SquareformOutput::Condensed(Array1::from_vec(out))
        }
    }
}

/// Compute a pairwise distance matrix between two sets of points.
pub fn rscdist(xa: &Array2<f64>, xb: &Array2<f64>, metric: &str) -> Array2<f64> {
    let (n_a, n_features) = xa.dim();
    let (n_b, _) = xb.dim();
    assert_eq!(n_features, xb.ncols(), "Feature counts must match");

    let mut out = Array2::zeros((n_a, n_b));
    for i in 0..n_a {
        let row_a = xa.row(i).to_slice().unwrap().to_vec();
        for j in 0..n_b {
            let row_b = xb.row(j).to_slice().unwrap();
            out[[i, j]] = match metric {
                "manhattan" | "cityblock" => manhattan(&row_a, row_b),
                "cosine" => cosine(&row_a, row_b),
                "chebyshev" => chebyshev(&row_a, row_b),
                _ => euclidean(&row_a, row_b),
            };
        }
    }
    out
}

// --- KDTree ---

#[derive(Clone, Debug)]
struct KDNode {
    point: Vec<f64>,
    index: usize,
    axis: usize,
    left: Option<Box<KDNode>>,
    right: Option<Box<KDNode>>,
}

/// K-d tree for fast spatial nearest-neighbor search.
pub struct KDTree {
    root: Option<Box<KDNode>>,
    pub data: Array2<f64>,
    pub n_features: usize,
}

impl KDTree {
    /// Build a KDTree from 2D points array (n_samples, n_features).
    pub fn new(data: &Array2<f64>) -> Self {
        let (n_samples, n_features) = data.dim();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        fn build_tree(
            indices: &mut [usize],
            data: &Array2<f64>,
            depth: usize,
            n_features: usize,
        ) -> Option<Box<KDNode>> {
            if indices.is_empty() {
                return None;
            }
            let axis = depth % n_features;
            indices.sort_by(|&a, &b| {
                data[[a, axis]]
                    .partial_cmp(&data[[b, axis]])
                    .unwrap_or(Ordering::Equal)
            });

            let mid = indices.len() / 2;
            let median_idx = indices[mid];
            let point = data.row(median_idx).to_slice().unwrap().to_vec();

            let left = build_tree(&mut indices[..mid], data, depth + 1, n_features);
            let right = build_tree(&mut indices[mid + 1..], data, depth + 1, n_features);

            Some(Box::new(KDNode {
                point,
                index: median_idx,
                axis,
                left,
                right,
            }))
        }

        let root = build_tree(&mut indices, data, 0, n_features);
        Self {
            root,
            data: data.clone(),
            n_features,
        }
    }

    /// Query the k nearest neighbors of a point.
    ///
    /// Returns `(distances, indices)` sorted in ascending order of distance.
    pub fn query(&self, point: &[f64], k: usize) -> (Vec<f64>, Vec<usize>) {
        assert_eq!(point.len(), self.n_features);
        let mut best: Vec<(f64, usize)> = Vec::new();

        fn search(node: &Option<Box<KDNode>>, point: &[f64], k: usize, best: &mut Vec<(f64, usize)>) {
            if let Some(n) = node {
                let d = euclidean(point, &n.point);
                best.push((d, n.index));
                best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                if best.len() > k {
                    best.pop();
                }

                let axis = n.axis;
                let diff = point[axis] - n.point[axis];

                let (first, second) = if diff < 0.0 {
                    (&n.left, &n.right)
                } else {
                    (&n.right, &n.left)
                };

                search(first, point, k, best);
                let worst_best = if best.len() < k {
                    f64::INFINITY
                } else {
                    best.last().unwrap().0
                };
                if diff.abs() < worst_best {
                    search(second, point, k, best);
                }
            }
        }

        search(&self.root, point, k, &mut best);
        let dists = best.iter().map(|&(d, _)| d).collect();
        let idxs = best.iter().map(|&(_, idx)| idx).collect();
        (dists, idxs)
    }

    /// Find all points within radius `r` of `point`.
    pub fn query_radius(&self, point: &[f64], r: f64) -> Vec<usize> {
        assert_eq!(point.len(), self.n_features);
        let mut results = Vec::new();

        fn search_r(node: &Option<Box<KDNode>>, point: &[f64], r: f64, results: &mut Vec<usize>) {
            if let Some(n) = node {
                let d = euclidean(point, &n.point);
                if d <= r {
                    results.push(n.index);
                }
                let axis = n.axis;
                let diff = point[axis] - n.point[axis];
                if diff - r <= 0.0 {
                    search_r(&n.left, point, r, results);
                }
                if diff + r >= 0.0 {
                    search_r(&n.right, point, r, results);
                }
            }
        }

        search_r(&self.root, point, r, &mut results);
        results.sort();
        results
    }
}

// --- ConvexHull (2D Monotone Chain) ---

/// Convex hull of 2D points.
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Indices of points on the convex hull in counter-clockwise order
    pub vertices: Vec<usize>,
    /// Edges (i, j) forming the perimeter
    pub simplice_edges: Vec<(usize, usize)>,
    /// Perimeter of the hull
    pub perimeter: f64,
    /// Area enclosed by the hull
    pub area: f64,
}

impl ConvexHull {
    /// Construct convex hull of 2D points (n_samples >= 3, 2).
    pub fn new(points: &Array2<f64>) -> Result<Self, String> {
        let (n, d) = points.dim();
        if n < 3 || d != 2 {
            return Err("ConvexHull requires at least 3 points in 2D".to_string());
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let cmp = points[[a, 0]].partial_cmp(&points[[b, 0]]).unwrap_or(Ordering::Equal);
            if cmp == Ordering::Equal {
                points[[a, 1]].partial_cmp(&points[[b, 1]]).unwrap_or(Ordering::Equal)
            } else {
                cmp
            }
        });

        // Cross product of OA and OB vectors: (A.x - O.x)*(B.y - O.y) - (A.y - O.y)*(B.x - O.x)
        let cross = |o: usize, a: usize, b: usize| -> f64 {
            (points[[a, 0]] - points[[o, 0]]) * (points[[b, 1]] - points[[o, 1]])
                - (points[[a, 1]] - points[[o, 1]]) * (points[[b, 0]] - points[[o, 0]])
        };

        // Lower hull
        let mut lower = Vec::new();
        for &p in &indices {
            while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-12 {
                lower.pop();
            }
            lower.push(p);
        }

        // Upper hull
        let mut upper = Vec::new();
        for &p in indices.iter().rev() {
            while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-12 {
                upper.pop();
            }
            upper.push(p);
        }

        lower.pop();
        upper.pop();
        lower.extend(upper);
        let hull = lower;

        let mut perimeter = 0.0;
        let mut area = 0.0;
        let mut simplice_edges = Vec::new();

        let h_len = hull.len();
        for i in 0..h_len {
            let p1 = hull[i];
            let p2 = hull[(i + 1) % h_len];
            simplice_edges.push((p1, p2));

            let x1 = points[[p1, 0]];
            let y1 = points[[p1, 1]];
            let x2 = points[[p2, 0]];
            let y2 = points[[p2, 1]];

            perimeter += ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
            area += x1 * y2 - x2 * y1;
        }

        area = (0.5 * area).abs();

        Ok(ConvexHull {
            vertices: hull,
            simplice_edges,
            perimeter,
            area,
        })
    }
}

// --- Delaunay Triangulation & Voronoi Diagrams (2D) ---

/// 2D Delaunay Triangulation computed via the Bowyer-Watson algorithm.
#[derive(Debug, Clone)]
pub struct Delaunay {
    pub points: Array2<f64>,
    /// Triangle indices [i, j, k]
    pub simplices: Vec<[usize; 3]>,
}

impl Delaunay {
    pub fn new(points: &Array2<f64>) -> Result<Self, String> {
        let (n, d) = points.dim();
        if n < 3 || d != 2 {
            return Err("Delaunay requires at least 3 points in 2D".to_string());
        }

        // Find bounding box
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for i in 0..n {
            let x = points[[i, 0]];
            let y = points[[i, 1]];
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }

        let dx = (max_x - min_x).max(1.0);
        let dy = (max_y - min_y).max(1.0);
        let delta = dx.max(dy) * 5.0;
        let mid_x = 0.5 * (min_x + max_x);
        let mid_y = 0.5 * (min_y + max_y);

        // Super-triangle vertices: indices n, n+1, n+2
        let p_st1 = [mid_x - 2.0 * delta, mid_y - delta];
        let p_st2 = [mid_x, mid_y + 2.0 * delta];
        let p_st3 = [mid_x + 2.0 * delta, mid_y - delta];

        let mut all_points = Vec::with_capacity(n + 3);
        for i in 0..n {
            all_points.push([points[[i, 0]], points[[i, 1]]]);
        }
        all_points.push(p_st1);
        all_points.push(p_st2);
        all_points.push(p_st3);

        let in_circumcircle = |pt: [f64; 2], v1: [f64; 2], v2: [f64; 2], v3: [f64; 2]| -> bool {
            let d = 2.0 * (v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) + v3[0] * (v1[1] - v2[1]));
            if d.abs() < 1e-14 {
                return false;
            }
            let ux = ((v1[0].powi(2) + v1[1].powi(2)) * (v2[1] - v3[1])
                + (v2[0].powi(2) + v2[1].powi(2)) * (v3[1] - v1[1])
                + (v3[0].powi(2) + v3[1].powi(2)) * (v1[1] - v2[1]))
                / d;
            let uy = ((v1[0].powi(2) + v1[1].powi(2)) * (v3[0] - v2[0])
                + (v2[0].powi(2) + v2[1].powi(2)) * (v1[0] - v3[0])
                + (v3[0].powi(2) + v3[1].powi(2)) * (v2[0] - v1[0]))
                / d;
            let r2 = (v1[0] - ux).powi(2) + (v1[1] - uy).powi(2);
            let dist2 = (pt[0] - ux).powi(2) + (pt[1] - uy).powi(2);
            dist2 <= r2 + 1e-12
        };

        let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

        for i in 0..n {
            let pt = all_points[i];
            let mut bad_triangles = Vec::new();

            for (t_idx, tri) in triangles.iter().enumerate() {
                if in_circumcircle(pt, all_points[tri[0]], all_points[tri[1]], all_points[tri[2]]) {
                    bad_triangles.push(t_idx);
                }
            }

            // Boundary polygon of bad triangles (edges that appear exactly once)
            let mut polygon_edges: Vec<(usize, usize)> = Vec::new();
            for &t_idx in &bad_triangles {
                let tri = triangles[t_idx];
                let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
                for (u, v) in edges {
                    let mut shared = false;
                    for &other_idx in &bad_triangles {
                        if other_idx != t_idx {
                            let otri = triangles[other_idx];
                            let oedges = [
                                (otri[0], otri[1]),
                                (otri[1], otri[0]),
                                (otri[1], otri[2]),
                                (otri[2], otri[1]),
                                (otri[2], otri[0]),
                                (otri[0], otri[2]),
                            ];
                            if oedges.contains(&(u, v)) {
                                shared = true;
                                break;
                            }
                        }
                    }
                    if !shared {
                        polygon_edges.push((u, v));
                    }
                }
            }

            // Remove bad triangles in reverse index order
            bad_triangles.sort_unstable_by(|a, b| b.cmp(a));
            for idx in bad_triangles {
                triangles.swap_remove(idx);
            }

            // Create new triangles to point i
            for (u, v) in polygon_edges {
                triangles.push([u, v, i]);
            }
        }

        // Remove any triangle containing super-triangle vertices
        let simplices: Vec<[usize; 3]> = triangles
            .into_iter()
            .filter(|tri| tri[0] < n && tri[1] < n && tri[2] < n)
            .collect();

        Ok(Delaunay {
            points: points.clone(),
            simplices,
        })
    }
}

/// 2D Voronoi diagram computed as the dual of Delaunay triangulation.
#[derive(Debug, Clone)]
pub struct Voronoi {
    /// Coordinates of Voronoi vertices (circumcenters of Delaunay triangles)
    pub vertices: Vec<[f64; 2]>,
    /// Delaunay triangulation used to build this Voronoi diagram
    pub delaunay: Delaunay,
    /// Ridge points: pairs of input point indices sharing an edge in the Delaunay graph
    pub ridge_points: Vec<(usize, usize)>,
}

impl Voronoi {
    pub fn new(points: &Array2<f64>) -> Result<Self, String> {
        let delaunay = Delaunay::new(points)?;
        let mut vertices = Vec::with_capacity(delaunay.simplices.len());

        // For each simplex, find circumcenter
        for tri in &delaunay.simplices {
            let v1 = [points[[tri[0], 0]], points[[tri[0], 1]]];
            let v2 = [points[[tri[1], 0]], points[[tri[1], 1]]];
            let v3 = [points[[tri[2], 0]], points[[tri[2], 1]]];

            let d = 2.0 * (v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) + v3[0] * (v1[1] - v2[1]));
            let (ux, uy) = if d.abs() < 1e-12 {
                ((v1[0] + v2[0] + v3[0]) / 3.0, (v1[1] + v2[1] + v3[1]) / 3.0)
            } else {
                let x = ((v1[0].powi(2) + v1[1].powi(2)) * (v2[1] - v3[1])
                    + (v2[0].powi(2) + v2[1].powi(2)) * (v3[1] - v1[1])
                    + (v3[0].powi(2) + v3[1].powi(2)) * (v1[1] - v2[1]))
                    / d;
                let y = ((v1[0].powi(2) + v1[1].powi(2)) * (v3[0] - v2[0])
                    + (v2[0].powi(2) + v2[1].powi(2)) * (v1[0] - v3[0])
                    + (v3[0].powi(2) + v3[1].powi(2)) * (v2[0] - v1[0]))
                    / d;
                (x, y)
            };
            vertices.push([ux, uy]);
        }

        // Collect unique ridge points from Delaunay edges
        let mut ridge_points = Vec::new();
        for tri in &delaunay.simplices {
            let edges = [
                (tri[0].min(tri[1]), tri[0].max(tri[1])),
                (tri[1].min(tri[2]), tri[1].max(tri[2])),
                (tri[0].min(tri[2]), tri[0].max(tri[2])),
            ];
            for e in edges {
                if !ridge_points.contains(&e) {
                    ridge_points.push(e);
                }
            }
        }

        Ok(Voronoi {
            vertices,
            delaunay,
            ridge_points,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distances() {
        let u = [1.0, 2.0, 3.0];
        let v = [4.0, 6.0, 3.0];

        assert_eq!(euclidean(&u, &v), 5.0);
        assert_eq!(manhattan(&u, &v), 7.0);
        assert_eq!(chebyshev(&u, &v), 4.0);
        assert!((minkowski(&u, &v, 2.0) - 5.0).abs() < 1e-9);
        assert!(cosine(&u, &v) > 0.0);
    }

    #[test]
    fn test_pdist_and_squareform() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let condensed = pdist(&x, "euclidean");
        assert_eq!(condensed.len(), 3);
        assert!((condensed[0] - 1.0).abs() < 1e-9); // d(0, 1) = 1.0
        assert!((condensed[1] - 1.0).abs() < 1e-9); // d(0, 2) = 1.0
        assert!((condensed[2] - 2.0_f64.sqrt()).abs() < 1e-9); // d(1, 2) = sqrt(2)

        match squareform(SquareformInput::Condensed(&condensed)) {
            SquareformOutput::Square(mat) => {
                assert_eq!(mat.dim(), (3, 3));
                assert_eq!(mat[[0, 0]], 0.0);
                assert!((mat[[0, 1]] - 1.0).abs() < 1e-9);
                assert!((mat[[1, 2]] - 2.0_f64.sqrt()).abs() < 1e-9);
            }
            _ => panic!("Expected square output"),
        }
    }

    #[test]
    fn test_kdtree() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 10.0, 10.0]).unwrap();
        let tree = KDTree::new(&data);

        let query_pt = [0.9, 0.9];
        let (_dists, idxs) = tree.query(&query_pt, 2);
        assert_eq!(idxs.len(), 2);
        assert_eq!(idxs[0], 1); // Point [1.0, 1.0] is nearest

        let in_radius = tree.query_radius(&[1.0, 1.0], 1.5);
        assert!(in_radius.contains(&0));
        assert!(in_radius.contains(&1));
        assert!(in_radius.contains(&2));
        assert!(!in_radius.contains(&3));
    }

    #[test]
    fn test_convexhull_2d() {
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5],
        )
        .unwrap();
        let hull = ConvexHull::new(&points).unwrap();
        assert_eq!(hull.vertices.len(), 4);
        assert!((hull.area - 1.0).abs() < 1e-6);
        assert!((hull.perimeter - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_delaunay_and_voronoi() {
        let points = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        )
        .unwrap();
        let del = Delaunay::new(&points).unwrap();
        assert_eq!(del.simplices.len(), 2);

        let vor = Voronoi::new(&points).unwrap();
        assert_eq!(vor.vertices.len(), 2);
        assert!(!vor.ridge_points.is_empty());
    }
}
