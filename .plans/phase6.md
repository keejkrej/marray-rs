# Phase 6: Optimization, Clustering, & Statistical Completeness

Expanding core optimization algorithms, hierarchical clustering, regression neighbors, hyperparameter search, and missing statistical tests.

## sp.optimize -- optimization and root finding

- [x] `brentq` -- Brent's method for 1D root finding
- [x] `newton` -- Newton-Raphson root finding
- [x] `minimize` (Nelder-Mead) -- derivative-free simplex optimization
- [x] `minimize` (BFGS) -- quasi-Newton optimization with inverse Hessian updates

## sp.stats -- statistical tests and distances

- [x] `chi2_contingency` -- Chi-square test of independence of variables in contingency table
- [x] `wasserstein_distance` -- 1D Earth Mover's Distance between distributions

## sklearn.cluster -- hierarchical clustering

- [x] `AgglomerativeClustering` -- hierarchical clustering with Ward, complete, average, single linkage

## sklearn.neighbors -- regression

- [x] `KNeighborsRegressor` -- k-nearest neighbors regression with uniform/distance weights

## sklearn.model_selection -- randomized search

- [x] `RandomizedSearchCV` -- randomized search on hyperparameters with cross-validation
