pub mod preprocessing;
pub mod linear_model;
pub mod cluster;
pub mod decomposition;
pub mod metrics;
pub mod model_selection;
pub mod tree;
pub mod naive_bayes;
pub mod ensemble;
pub mod neighbors;
pub mod svm;
pub mod neural_network;
pub mod pipeline;
pub mod impute;
pub mod feature_selection;
pub mod discriminant_analysis;
pub mod mixture;
pub mod gaussian_process;
pub mod manifold;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Axis};

    #[test]
    fn test_standard_scaler() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = preprocessing::StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x);

        let mean = x_scaled.mean_axis(Axis(0)).unwrap();
        assert!(mean[0].abs() < 1e-9);
        assert!(mean[1].abs() < 1e-9);
    }

    #[test]
    fn test_linear_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut reg = linear_model::LinearRegression::new(true);
        reg.fit(&x, &y).unwrap();

        let coef = reg.coef.as_ref().unwrap();
        let intercept = reg.intercept.unwrap();

        assert!((coef[0] - 2.0).abs() < 1e-9);
        assert!((intercept - 1.0).abs() < 1e-9);

        let preds = reg.predict(&array![[5.0]]);
        assert!((preds[0] - 11.0).abs() < 1e-9);
    }

    #[test]
    fn test_ridge_regression() {
        // y = 2*x + 1
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut reg = linear_model::Ridge::new(0.1, true);
        reg.fit(&x, &y).unwrap();

        let coef = reg.coef.as_ref().unwrap();
        let intercept = reg.intercept.unwrap();

        // L2 regularization shrinks coefficients slightly towards 0
        assert!((coef[0] - 2.0).abs() < 0.1);
        assert!((intercept - 1.0).abs() < 0.2);

        let preds = reg.predict(&array![[5.0]]);
        assert!((preds[0] - 11.0).abs() < 0.5);
    }

    #[test]
    fn test_logistic_regression() {
        let x = array![[0.1], [0.2], [0.8], [0.9]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let mut clf = linear_model::LogisticRegression::new(1000, 0.5, 1.0);
        clf.fit(&x, &y);

        let preds = clf.predict(&array![[0.15], [0.85]]);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_kmeans() {
        let x = array![[1.0, 1.0], [1.5, 1.5], [10.0, 10.0], [10.5, 10.5]];
        let mut km = cluster::KMeans::new(2, 100);
        km.fit(&x);
        let preds = km.predict(&x);
        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[2], preds[3]);
        assert_ne!(preds[0], preds[2]);
    }

    #[test]
    fn test_pca() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let mut pca = decomposition::PCA::new(1);
        pca.fit(&x).unwrap();
        let transformed = pca.transform(&x);
        assert_eq!(transformed.dim(), (3, 1));
    }

    #[test]
    fn test_train_test_split() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (x_train, x_test, y_train, y_test) = model_selection::train_test_split(&x, &y, 0.4, false);
        assert_eq!(x_train.nrows(), 3);
        assert_eq!(x_test.nrows(), 2);
        assert_eq!(y_train.len(), 3);
        assert_eq!(y_test.len(), 2);
    }

    #[test]
    fn test_decision_tree() {
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0],
            [11.0, 10.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut clf = tree::DecisionTreeClassifier::new(3, 2);
        clf.fit(&x, &y);

        let preds = clf.predict(&array![[1.5, 1.5], [10.5, 10.5]]);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_one_hot_encoder() {
        let x = array![
            [0.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0]
        ];
        let mut encoder = preprocessing::OneHotEncoder::new();
        let encoded = encoder.fit_transform(&x);
        assert_eq!(encoded.dim(), (3, 4));
        assert_eq!(encoded[[0, 0]], 1.0);
        assert_eq!(encoded[[0, 1]], 0.0);
        assert_eq!(encoded[[0, 2]], 1.0);
        assert_eq!(encoded[[0, 3]], 0.0);
    }

    #[test]
    fn test_gaussian_nb() {
        let x = array![
            [1.0, 1.0],
            [1.2, 0.9],
            [8.0, 8.0],
            [8.5, 8.2]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let mut clf = naive_bayes::GaussianNB::new();
        clf.fit(&x, &y);
        let preds = clf.predict(&array![[1.1, 1.0], [8.2, 8.1]]);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_bayesian_linear_regression_known_variance() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut model = linear_model::BayesianLinearRegressionKnownVariance::new(0.0, 1.0, None, true);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[5.0]]);

        assert_eq!(pred.len(), 1);
        assert!(pred[0] > 10.0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_bayesian_linear_regression_unknown_variance() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut model = linear_model::BayesianLinearRegressionUnknownVariance::new(1.0, 1.0, 0.0, None, true);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[5.0]]);

        assert_eq!(pred.len(), 1);
        assert!(pred[0] > 10.0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_generalized_linear_model_identity_link() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mut model = linear_model::GeneralizedLinearModel::new("identity", true, 1e-6, 25);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[4.0]]);

        assert_eq!(pred.len(), 1);
        assert!((pred[0] - 5.0).abs() < 0.2);
    }

    #[test]
    fn test_pipeline_standard_scaler_knn() {
        use pipeline::{make_pipeline, StandardScalerWrapper};
        use neighbors::KNeighborsClassifier;

        let x_train = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let scaler = StandardScalerWrapper::new();
        let mut pipe = make_pipeline(vec![Box::new(scaler)]);
        let x_scaled = pipe.fit_transform(&x_train);

        let mean = x_scaled.mean_axis(Axis(0)).unwrap();
        assert!(mean[0].abs() < 1e-9);
        assert!(mean[1].abs() < 1e-9);

        let mut clf = KNeighborsClassifier::new(3);
        clf.fit(&x_scaled, &y_train);
        let x_test = pipe.transform(&array![[0.5, 0.5], [10.5, 10.5]]);
        let preds = clf.predict(&x_test);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_simple_imputer_mean() {
        use impute::SimpleImputer;
        use ndarray::array;

        let x = array![
            [1.0, f64::NAN],
            [2.0, 3.0],
            [f64::NAN, 5.0],
            [4.0, 7.0],
        ];
        let mut imputer = SimpleImputer::new("mean", 0.0);
        let result = imputer.fit_transform(&x);

        assert!((result[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((result[[0, 1]] - 5.0).abs() < 1e-9);
        assert!((result[[1, 0]] - 2.0).abs() < 1e-9);
        assert!((result[[1, 1]] - 3.0).abs() < 1e-9);
        assert!((result[[2, 0]] - 7.0 / 3.0).abs() < 1e-9);
        assert!((result[[2, 1]] - 5.0).abs() < 1e-9);
        assert!((result[[3, 0]] - 4.0).abs() < 1e-9);
        assert!((result[[3, 1]] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_dbscan_two_clusters() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];
        let dbscan = cluster::DBSCAN::new(1.0, 2);
        let labels = dbscan.fit_predict(&x);

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_dbscan_noise() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [50.0, 50.0],
        ];
        let dbscan = cluster::DBSCAN::new(0.5, 2);
        let labels = dbscan.fit_predict(&x);

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], -1.0);
    }
}
