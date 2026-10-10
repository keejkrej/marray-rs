use mlab::np;
use mlab::sklearn::{
    cluster::KMeans,
    ensemble::RandomForestClassifier,
    decomposition::PCA,
    linear_model::{LinearRegression, LogisticRegression, Ridge},
    metrics::{accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score},
    model_selection::{cross_val_score, train_test_split},
    naive_bayes::GaussianNB,
    neighbors::KNeighborsClassifier,
    preprocessing::{LabelEncoder, OneHotEncoder, StandardScaler},
    tree::DecisionTreeClassifier,
};

fn main() {
    println!("--- Scikit-Learn Demo ---");

    // 1. Preprocessing and Linear Regression
    // Let's create some dummy regression data: y = 3 * x + 5 + noise
    let x = np::array(vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![5.0],
        vec![6.0],
        vec![7.0],
        vec![8.0],
    ]);
    let y = np::array(vec![8.1, 10.9, 14.2, 17.0, 20.1, 22.9, 26.2, 29.0]);

    // Split data
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.25, false);

    // Fit StandardScaler
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(&x_train);
    let x_test_scaled = scaler.transform(&x_test);

    // Fit LinearRegression
    let mut reg = LinearRegression::new(true);
    reg.fit(&x_train_scaled, &y_train).unwrap();

    let preds = reg.predict(&x_test_scaled);
    println!("Linear Regression coefficients: {:?}", reg.coef);
    println!("Linear Regression intercept: {:?}", reg.intercept);
    println!("Predictions on test set: {:?}", preds);
    println!("True test labels: {:?}", y_test);
    println!("MSE: {}", mean_squared_error(&y_test, &preds));
    println!("R2 Score: {}", r2_score(&y_test, &preds));
    println!();

    // Fit Ridge Regression
    let mut ridge = Ridge::new(0.5, true);
    ridge.fit(&x_train_scaled, &y_train).unwrap();

    let ridge_preds = ridge.predict(&x_test_scaled);
    println!("Ridge Regression coefficients: {:?}", ridge.coef);
    println!("Ridge Regression intercept: {:?}", ridge.intercept);
    println!("Ridge Predictions on test set: {:?}", ridge_preds);
    println!("Ridge MSE: {}", mean_squared_error(&y_test, &ridge_preds));
    println!("Ridge R2 Score: {}", r2_score(&y_test, &ridge_preds));
    println!();

    // 2. Logistic Regression
    let x_clf = np::array(vec![
        vec![0.1],
        vec![0.25],
        vec![0.3],
        vec![0.7],
        vec![0.85],
        vec![0.9],
    ]);
    let y_clf = np::array(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut clf = LogisticRegression::new(1000, 0.1, 1.0);
    clf.fit(&x_clf, &y_clf);
    let clf_preds = clf.predict(&x_clf);
    println!("Logistic Regression Predictions: {:?}", clf_preds);
    println!("Accuracy Score: {}", accuracy_score(&y_clf, &clf_preds));
    println!();

    // 3. KMeans Clustering
    let x_cluster = np::array(vec![
        vec![1.0, 1.0],
        vec![1.2, 0.8],
        vec![10.0, 10.0],
        vec![9.8, 10.2],
    ]);
    let mut kmeans = KMeans::new(2, 100);
    kmeans.fit(&x_cluster);
    println!("KMeans Centers:\n{:?}", kmeans.cluster_centers);
    println!("KMeans Predictions: {:?}", kmeans.predict(&x_cluster));
    println!();

    // 4. PCA
    let x_pca = np::array(vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
    ]);
    let mut pca = PCA::new(1);
    pca.fit(&x_pca).unwrap();
    let transformed = pca.transform(&x_pca);
    println!("PCA Components:\n{:?}", pca.components);
    println!("PCA Explained Variance: {:?}", pca.explained_variance);
    println!("PCA Transformed Data:\n{:?}", transformed);
    println!();

    // 5. Decision Tree Classifier
    let x_tree = np::array(vec![
        vec![1.0, 1.0],
        vec![1.5, 1.2],
        vec![2.0, 2.0],
        vec![10.0, 10.0],
        vec![10.5, 9.8],
        vec![11.0, 11.0],
    ]);
    let y_tree = np::array(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let mut tree_clf = DecisionTreeClassifier::new(3, 2);
    tree_clf.fit(&x_tree, &y_tree);
    let tree_preds = tree_clf.predict(&x_tree);
    println!("Decision Tree Predictions: {:?}", tree_preds);
    println!("Decision Tree Accuracy: {}", accuracy_score(&y_tree, &tree_preds));
    println!();

    // 6. OneHotEncoder
    let x_cat = np::array(vec![
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![0.0, 2.0],
        vec![2.0, 0.0],
    ]);
    println!("Original categorical features:\n{:?}", x_cat);
    let mut encoder = OneHotEncoder::new();
    let encoded = encoder.fit_transform(&x_cat);
    println!("One-hot encoded representation:\n{:?}", encoded);
    println!("Categories per column: {:?}", encoder.categories);
    println!();

    // 7. New helpers: KNN, Random Forest, metrics, cross_val_score, LabelEncoder
    let mut knn = KNeighborsClassifier::new(1);
    knn.fit(&np::array(vec![vec![0.0], vec![1.0], vec![2.0]]), &np::array(vec![0.0, 1.0, 1.0]));
    println!("KNN prediction: {:?}", knn.predict(&np::array(vec![vec![0.9]])));

    let mut rf = RandomForestClassifier::new(3);
    rf.fit(&np::array(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]), &np::array(vec![0.0, 0.0, 1.0, 1.0]));
    println!("RandomForest prediction: {:?}", rf.predict(&np::array(vec![vec![1.5]])));

    let y_true = np::array(vec![0.0, 1.0, 1.0, 0.0]);
    let y_pred = np::array(vec![0.0, 0.0, 1.0, 1.0]);
    println!("Confusion matrix: {:?}", confusion_matrix(&y_true, &y_pred));
    println!("Classification report: {:?}", classification_report(&y_true, &y_pred));

    struct DummyScore;
    impl mlab::sklearn::model_selection::Score for DummyScore {
        fn score(&self, _x: &mlab::np::Array2<f64>, _y: &mlab::np::Array1<f64>) -> f64 { 1.0 }
    }
    println!("Cross-val score: {:?}", cross_val_score(&DummyScore, &np::array(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]), &np::array(vec![0.0, 0.0, 1.0, 1.0]), 2));

    let mut encoder = LabelEncoder::new();
    let encoded = encoder.fit_transform(&np::array(vec![2.0, 1.0, 2.0, 0.0]));
    println!("LabelEncoder classes: {:?}", encoder.classes);
    println!("LabelEncoder encoded labels: {:?}", encoded);

    // 8. Gaussian Naive Bayes Classifier
    let x_nb = np::array(vec![
        vec![1.0, 2.0],
        vec![1.2, 1.8],
        vec![1.5, 2.2],
        vec![6.0, 8.0],
        vec![6.5, 7.5],
        vec![7.0, 8.5],
    ]);
    let y_nb = np::array(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let mut gnb = GaussianNB::new();
    gnb.fit(&x_nb, &y_nb);

    let test_nb = np::array(vec![
        vec![1.1, 1.9],
        vec![6.8, 8.2],
    ]);
    let nb_preds = gnb.predict(&test_nb);
    println!("Test samples:\n{:?}", test_nb);
    println!("Gaussian NB Predictions: {:?}", nb_preds);
    println!();
}
