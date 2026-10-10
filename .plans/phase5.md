# Phase 5: Advanced & Niche

Advanced scientific computing and specialized algorithms.

## sp.integrate -- numerical integration

- [x] `quad` -- 1D quadrature (adaptive Gauss-Kronrod / Simpson)
- [x] `dblquad` -- 2D quadrature
- [x] `trapz` / `simpson` -- trapezoidal/Simpson's rule
- [x] `cumtrapz` -- cumulative trapezoidal
- [x] `solve_ivp` -- ODE initial value solver (RK45, RK23, Radau, BDF)
- [x] `odeint` -- ODE integrator (scipy-compatible)
- [x] `romberg` -- Romberg integration

## sp.sparse -- sparse matrix operations

- [x] `csr_matrix` -- compressed sparse row
- [x] `csc_matrix` -- compressed sparse column
- [x] `coo_matrix` -- coordinate format
- [x] Sparse arithmetic (+, -, *, dot)
- [x] `linalg.spsolve` -- sparse linear solve
- [x] `linalg.eigsh` -- sparse eigenvalue solver
- [x] `linalg.norm` -- sparse norm

## sp.spatial -- spatial data structures

- [x] `KDTree` -- k-d tree for nearest neighbor
- [x] `pdist` -- pairwise distances (condensed form)
- [x] `squareform` -- convert between condensed and square
- [x] `ConvexHull` -- convex hull
- [x] `Voronoi` -- Voronoi tessellation
- [x] `Delaunay` -- Delaunay triangulation
- [x] Distance functions: `euclidean`, `manhattan`, `cosine`, `chebyshev`, `minkowski`

## sklearn.manifold -- dimensionality reduction

- [x] `TSNE` -- t-distributed stochastic neighbor embedding
- [x] `Isomap` -- isometric mapping

## sklearn.feature_selection -- feature selection

- [x] `SelectKBest` -- select top k features
- [x] `RFE` -- recursive feature elimination
- [x] `mutual_info_classif` / `mutual_info_reg` -- mutual information

## sklearn.gaussian_process -- Gaussian processes

- [x] `GaussianProcessRegressor` -- GP regression
- [x] `GaussianProcessClassifier` -- GP classification

## sklearn.mixture -- mixture models

- [x] `GaussianMixture` -- Gaussian mixture model (sklearn interface)

## sklearn.discriminant_analysis

- [x] `LinearDiscriminantAnalysis` -- LDA
- [x] `QuadraticDiscriminantAnalysis` -- QDA

## sklearn.decomposition -- additional methods

- [x] `NMF` -- non-negative matrix factorization
- [x] `TruncatedSVD` -- truncated SVD (for sparse)
- [x] `FactorAnalysis` -- factor analysis

## sp.special -- additional special functions

- [x] `iv`, `jv` -- Bessel functions (arbitrary order)
- [x] `kv`, `yv` -- modified Bessel of 2nd kind, Bessel of 2nd kind
- [x] `ndtr` / `ndtri` -- normal CDF and its inverse
- [x] `hyp1f1`, `hyp2f1` -- hypergeometric functions
- [x] `ellipk`, `ellipe` -- complete elliptic integrals
- [x] `airy` -- Airy functions
- [x] `sph_harm` -- spherical harmonics
