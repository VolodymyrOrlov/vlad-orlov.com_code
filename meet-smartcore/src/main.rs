use std::fs::File;
use std::io::prelude::*;

use nalgebra::{DMatrix, RowDVector};

use ndarray::Array;

use smartcore::dataset::boston;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::error::Failed;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::mean_absolute_error;
use smartcore::model_selection::{cross_validate, train_test_split, KFold};

fn load_bos_dataset() -> (DMatrix<f32>, RowDVector<f32>) {
    // Load The Boston Housing dataset
    let data = boston::load_dataset();
    // Turn Boston Housing dataset into nalgebra matrix
    let x = DMatrix::from_row_slice(data.num_samples, data.num_features, &data.data);
    // These are our target values
    let y = RowDVector::from_iterator(data.num_samples, data.target.into_iter());

    (x, y)
}

fn linear_regression() -> Result<(), Failed> {
    let (x, y) = load_bos_dataset();
    // Split data into training/test sets
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    // Fit logistic regression and predict value estimates using test set
    let y_hat = LinearRegression::fit(&x_train, &y_train, Default::default())
        .and_then(|lr| lr.predict(&x_test))?;
    // Validate model performance on a test set
    println!(
        "Linear Regression MAE: {}",
        mean_absolute_error(&y_test, &y_hat)
    );
    Ok(())
}

fn random_forest() -> Result<(), Failed> {
    let (x, y) = load_bos_dataset();
    // Split data into training/test sets
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    // Fit logistic regression and predict value estimates using test set
    let y_hat = RandomForestRegressor::fit(&x_train, &y_train, Default::default())
        .and_then(|lr| lr.predict(&x_test))?;
    // Validate model performance on a test set
    println!(
        "Random Forest MAE: {}",
        mean_absolute_error(&y_test, &y_hat)
    );
    Ok(())
}

fn cv_random_forest() -> Result<(), Failed> {
    let (x, y) = load_bos_dataset();
    // cross-validate our model
    let results = cross_validate(
        RandomForestRegressor::fit,
        &x,
        &y,
        Default::default(),                // hyperparameters
        KFold::default().with_n_splits(3), // 3-fold split
        mean_absolute_error,
    )?;
    println!("Random Forest CV MAE: {}", results.mean_test_score());
    Ok(())
}

fn save_random_forest_model() -> Result<(), Failed> {
    let (x, y) = load_bos_dataset();
    // Train the model
    let model = RandomForestRegressor::fit(&x, &y, Default::default())?;
    // Save the model
    let model_bytes = bincode::serialize(&model).expect("Can not serialize the model");
    File::create("random_forest.model")
        .and_then(|mut f| f.write_all(&model_bytes))
        .expect("Can not persist the model");
    Ok(())
}

fn linear_regression_ndarray() -> Result<(), Failed> {
    let data = boston::load_dataset();
    // and turn data into a NxM matrix
    let x = Array::from_shape_vec((data.num_samples, data.num_features), data.data).unwrap();

    let y = Array::from_shape_vec(data.num_samples, data.target).unwrap();

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

    let y_hat = LinearRegression::fit(&x_train, &y_train, Default::default())
        .and_then(|lr| lr.predict(&x_test))?;

    println!(
        "Linear Regression, ndarray, MAE: {}",
        mean_absolute_error(&y_test, &y_hat)
    );

    Ok(())
}

fn main() {
    linear_regression().unwrap();
    random_forest().unwrap();
    cv_random_forest().unwrap();
    save_random_forest_model().unwrap();
    // bonus
    linear_regression_ndarray().unwrap();
}
