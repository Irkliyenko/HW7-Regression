"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# Load data
X_train, X_val, y_train, y_val = utils.loadDataset(
	features=[
		'Penicillin V Potassium 500 MG',
		'Computed tomography of chest and abdomen',
		'Plain chest X-ray (procedure)',
		'Low Density Lipoprotein Cholesterol',
		'Creatinine',
		'AGE_DIAGNOSIS'
	],
	split_percent=0.8,
	split_seed=42
)

# Scale the data, since values vary across feature. Note that we
# fit on the training data and use the same scaler for X_val.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

def test_prediction():
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.0001, max_iter=300, batch_size=10, reg_type="l2", lambda_=0.01)
    log_model.train_model(X_train, y_train, X_val, y_val)

    # Get predictions from our custom model (train and validation sets)
    y_pred_train = log_model.make_prediction(X_train)
    y_pred_val = log_model.make_prediction(X_val)

    # Convert probabilities to binary labels (threshold = 0.5)
    y_pred_train = (y_pred_train >= 0.5).astype(int)
    y_pred_val = (y_pred_val >= 0.5).astype(int)

    # Ensure predictions are within valid range
    assert set(np.unique(y_pred_train)).issubset({0, 1}), "Train predictions have invalid values"
    assert set(np.unique(y_pred_val)).issubset({0, 1}), "Validation predictions have invalid values"

    # Check similarity
    train_match = (y_pred_train == y_train).mean()
    val_match = (y_pred_val == y_val).mean()
    assert train_match > 0.7, f"Train predictions are not sufficiently close, match={train_match}"
    assert val_match > 0.7, f"Validation predictions are not sufficiently close, match={val_match}"
	

def test_loss_function():
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.0001, max_iter=300, batch_size=10, reg_type="l2", lambda_=0.01)
    log_model.train_model(X_train, y_train, X_val, y_val)

    # Get predictions
    y_pred_train = log_model.make_prediction(X_train)
    y_pred_val = log_model.make_prediction(X_val)

    # Compute loss
    loss_train = log_model.loss_function(y_train, y_pred_train)
    loss_val = log_model.loss_function(y_val, y_pred_val)

    # Ensure loss is non-negative
    assert loss_train >= 0, "Train loss should be non-negative"
    assert loss_val >= 0, "Validation loss should be non-negative"

    # Ensure loss is not NaN
    assert not np.isnan(loss_train), "Train loss should not be NaN"
    assert not np.isnan(loss_val), "Validation loss should not be NaN"

def test_gradient():
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.0001, max_iter=300, batch_size=10, reg_type="l2", lambda_=0.01) 
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    # Compute gradient using the model
    grad = log_model.calculate_gradient(y_train, X_train, reg_type="l2", lambda_=0.01)

    # Ensure gradient is not NaN
    assert not np.isnan(grad).any(), "Gradient contains NaN values"
    assert not np.isinf(grad).any(), "Gradient contains Inf values"


def test_training():
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.01, tol=0.0001, max_iter=300, batch_size=10, reg_type="l2", lambda_=0.01)

    # Save initial weights
    initial_W = log_model.W.copy()

    # Train the model
    log_model.train_model(X_train, y_train, X_val, y_val)

    # Ensure weights have changed
    assert not np.allclose(log_model.W, initial_W), "Weights did not update during training"

    # Ensure loss decreases over training
    assert log_model.loss_hist_train[0] > log_model.loss_hist_train[-1], "Loss did not decrease over training"
