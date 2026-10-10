pub mod dt;
pub mod gbdt;
pub mod losses;
pub mod rf;

pub use dt::DecisionTree;
pub use gbdt::GradientBoostedDecisionTree;
pub use losses::{CrossEntropyLoss, MSELoss, TreeLoss};
pub use rf::RandomForest;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_decision_tree_classifier() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut dt = DecisionTree::new(true, Some(3), None, Some("gini"), Some(42));
        dt.fit(&x, &y);

        let preds = dt.predict(&x);
        assert_eq!(preds, y);

        let test_x = array![[0.05, 0.05], [1.05, 1.05]];
        let test_preds = dt.predict(&test_x);
        assert_eq!(test_preds[0], 0.0);
        assert_eq!(test_preds[1], 1.0);
    }

    #[test]
    fn test_decision_tree_regressor() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];

        let mut dt = DecisionTree::new(false, Some(4), None, Some("mse"), Some(42));
        dt.fit(&x, &y);

        let preds = dt.predict(&x);
        let mse = (&preds - &y).iter().map(|&v| v * v).sum::<f64>() / y.len() as f64;
        assert!(mse < 0.1, "DecisionTree regression MSE: {}", mse);
    }

    #[test]
    fn test_random_forest() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut rf = RandomForest::new(5, Some(3), None, true, Some("entropy"), Some(42));
        rf.fit(&x, &y);

        let test_x = array![[0.05, 0.05], [10.05, 10.05]];
        let preds = rf.predict(&test_x);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_gbdt_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let mut gbdt = GradientBoostedDecisionTree::new(10, Some(2), false, 0.5, Some("mse"));
        gbdt.fit(&x, &y);

        let preds = gbdt.predict(&x);
        let mse = (&preds - &y).iter().map(|&v| v * v).sum::<f64>() / y.len() as f64;
        assert!(mse < 0.5, "GBDT regression MSE: {}", mse);
    }

    #[test]
    fn test_loss_functions() {
        let y_true = array![1.0, 0.0];
        let y_pred = array![0.8, 0.2];

        let mse_loss = MSELoss::new();
        let mse = mse_loss.loss(&y_true, &y_pred);
        assert!((mse - 0.04).abs() < 1e-6);

        let ce_loss = CrossEntropyLoss::new();
        let ce = ce_loss.loss(&y_true, &y_pred);
        assert!(ce > 0.0);
    }
}
