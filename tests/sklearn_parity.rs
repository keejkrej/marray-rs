use mlab::sklearn;
use ndarray::{array, Axis};

#[test]
fn test_sklearn_preprocessing_parity() {
    let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];

    // StandardScaler
    let mut scaler = sklearn::preprocessing::StandardScaler::new();
    let x_scaled = scaler.fit_transform(&x);
    let mean = x_scaled.mean_axis(Axis(0)).unwrap();
    assert!(mean[0].abs() < 1e-7);
    assert!(mean[1].abs() < 1e-7);
    assert!(mean[2].abs() < 1e-7);

    // MinMaxScaler
    let mut minmax = sklearn::preprocessing::MinMaxScaler::new();
    let x_minmax = minmax.fit_transform(&x);
    for v in x_minmax.iter() {
        assert!(*v >= -1e-7 && *v <= 1.0 + 1e-7);
    }

    // LabelEncoder
    let mut le = sklearn::preprocessing::LabelEncoder::new();
    let labels = array![10.0, 20.0, 10.0, 30.0];
    let encoded = le.fit_transform(&labels);
    assert_eq!(encoded.len(), 4);
    assert_eq!(encoded[0], encoded[2]); // 10.0 == 10.0
    assert_ne!(encoded[0], encoded[1]); // 10.0 != 20.0

    // SimpleImputer
    let mut imputer = sklearn::impute::SimpleImputer::new("mean", f64::NAN);
    let data_with_nan = array![[1.0, 2.0], [f64::NAN, 4.0], [3.0, 6.0]];
    let imputed = imputer.fit_transform(&data_with_nan);
    assert!(!imputed[[1, 0]].is_nan());
    assert!((imputed[[1, 0]] - 2.0).abs() < 1e-6); // mean of 1.0 and 3.0 is 2.0
}

#[test]
fn test_sklearn_linear_and_trees_parity() {
    // LinearRegression
    let x_reg = array![[1.0], [2.0], [3.0], [4.0]];
    let y_reg = array![2.0, 4.0, 6.0, 8.0];
    let mut lr = sklearn::linear_model::LinearRegression::new(true);
    lr.fit(&x_reg, &y_reg).unwrap();
    let preds_lr = lr.predict(&array![[5.0]]);
    assert!((preds_lr[0] - 10.0).abs() < 1e-6);

    // Ridge regression
    let mut ridge = sklearn::linear_model::Ridge::new(1.0, true);
    ridge.fit(&x_reg, &y_reg).unwrap();
    let preds_ridge = ridge.predict(&array![[5.0]]);
    assert!((preds_ridge[0] - 9.1666667).abs() < 1e-4);

    // LogisticRegression
    let x_clf = array![[0.0], [0.5], [1.5], [2.0]];
    let y_clf = array![0.0, 0.0, 1.0, 1.0];
    let mut logreg = sklearn::linear_model::LogisticRegression::new(500, 0.1, 1.0);
    logreg.fit(&x_clf, &y_clf);
    let preds_logreg = logreg.predict(&array![[0.1], [1.9]]);
    assert_eq!(preds_logreg[0], 0.0);
    assert_eq!(preds_logreg[1], 1.0);

    // DecisionTreeClassifier
    let mut dt = sklearn::tree::DecisionTreeClassifier::new(3, 2);
    dt.fit(&x_clf, &y_clf);
    let dt_preds = dt.predict(&x_clf);
    assert_eq!(dt_preds, y_clf);

    // DecisionTreeRegressor
    let mut dtr = sklearn::tree::DecisionTreeRegressor::new(4, 2, 1);
    dtr.fit(&x_reg, &y_reg);
    let dtr_preds = dtr.predict(&x_reg);
    let mse = sklearn::metrics::mean_squared_error(&y_reg, &dtr_preds);
    assert!(mse < 0.1);
}

#[test]
fn test_sklearn_ensemble_and_neighbors_parity() {
    let x = array![
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
        [1.0, 1.0], [1.1, 0.9], [0.9, 1.1]
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    // KNeighborsClassifier
    let mut knn = sklearn::neighbors::KNeighborsClassifier::new(3);
    knn.fit(&x, &y);
    let knn_preds = knn.predict(&array![[0.05, 0.05], [0.95, 0.95]]);
    assert_eq!(knn_preds[0], 0.0);
    assert_eq!(knn_preds[1], 1.0);

    // KNeighborsRegressor
    let y_reg = array![0.0, 0.1, 0.2, 2.0, 2.1, 1.9];
    let mut knr = sklearn::neighbors::KNeighborsRegressor::new(3, Some("uniform"));
    knr.fit(&x, &y_reg);
    let knr_preds = knr.predict(&array![[0.0, 0.0], [1.0, 1.0]]);
    assert!(knr_preds[0] < 0.5);
    assert!(knr_preds[1] > 1.5);

    // RandomForestClassifier
    let mut rf = sklearn::ensemble::RandomForestClassifier::new(10);
    rf.fit(&x, &y);
    let rf_preds = rf.predict(&x);
    assert_eq!(rf_preds, y);

    // GradientBoostingClassifier
    let mut gbc = sklearn::ensemble::GradientBoostingClassifier::new(10, 0.2, 2);
    gbc.fit(&x, &y);
    let gbc_preds = gbc.predict(&x);
    assert_eq!(gbc_preds, y);
}

#[test]
fn test_sklearn_cluster_and_manifold_parity() {
    let x = array![
        [1.0, 1.0], [1.2, 1.1], [0.9, 0.8],
        [10.0, 10.0], [10.2, 9.9], [9.8, 10.1]
    ];

    // KMeans
    let mut km = sklearn::cluster::KMeans::new(2, 50);
    km.fit(&x);
    let labels_km = km.predict(&x);
    assert_eq!(labels_km[0], labels_km[1]);
    assert_eq!(labels_km[3], labels_km[4]);
    assert_ne!(labels_km[0], labels_km[3]);

    // DBSCAN
    let db = sklearn::cluster::DBSCAN::new(2.0, 2);
    let labels_db = db.fit_predict(&x);
    assert_eq!(labels_db[0], labels_db[1]);
    assert_eq!(labels_db[3], labels_db[4]);
    assert_ne!(labels_db[0], labels_db[3]);

    // AgglomerativeClustering
    let agg = sklearn::cluster::AgglomerativeClustering::new(2, Some("ward"));
    let labels_agg = agg.fit_predict(&x);
    assert_eq!(labels_agg[0], labels_agg[1]);
    assert_eq!(labels_agg[3], labels_agg[4]);
    assert_ne!(labels_agg[0], labels_agg[3]);

    // PCA
    let mut pca = sklearn::decomposition::PCA::new(1);
    let transformed = pca.fit_transform(&x).unwrap();
    assert_eq!(transformed.shape(), &[6, 1]);

    // TSNE
    let mut tsne = sklearn::manifold::TSNE::new(2, Some(5.0), Some(50.0), Some(250));
    let embedded = tsne.fit_transform(&x).unwrap();
    assert_eq!(embedded.shape(), &[6, 2]);
}

#[test]
fn test_sklearn_metrics_and_model_selection_parity() {
    let y_true = array![0.0, 1.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.0, 1.0, 0.0, 0.0, 1.0];

    let acc = sklearn::metrics::accuracy_score(&y_true, &y_pred);
    assert_eq!(acc, 0.8);

    let prec = sklearn::metrics::precision_score(y_true.as_slice().unwrap(), y_pred.as_slice().unwrap(), "binary");
    assert_eq!(prec, 1.0); // 2 TP, 0 FP

    let rec = sklearn::metrics::recall_score(y_true.as_slice().unwrap(), y_pred.as_slice().unwrap(), "binary");
    assert!((rec - (2.0 / 3.0)).abs() < 1e-6);

    let cm = sklearn::metrics::confusion_matrix(&y_true, &y_pred);
    assert_eq!(cm[[0, 0]], 2); // TN
    assert_eq!(cm[[1, 1]], 2); // TP

    // train_test_split
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]];
    let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let (x_tr, x_te, y_tr, y_te) = sklearn::model_selection::train_test_split(&x, &y, 0.3, false);
    assert_eq!(x_tr.nrows(), 7);
    assert_eq!(x_te.nrows(), 3);
    assert_eq!(y_tr.len(), 7);
    assert_eq!(y_te.len(), 3);

    // KFold
    let kf = sklearn::model_selection::KFold::new(5, false, None);
    let splits = kf.split(10);
    assert_eq!(splits.len(), 5);
    for (tr, te) in splits {
        assert_eq!(tr.len(), 8);
        assert_eq!(te.len(), 2);
    }
}
