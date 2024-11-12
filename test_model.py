import pytest
import numpy as np
from ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data():
    # Create sample data for testing using numpy arrays 
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[2, 3, 4], [8, 9, 10]])
    y_test = np.array([0, 1])
    return X_train, y_train, X_test, y_test

def test_train_model(sample_data):
    X_train, y_train, _, _ = sample_data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "The model should be a RandomForestClassifier instance"

def test_compute_model_metrics(sample_data):
    _, _, X_test, y_test = sample_data
    model = RandomForestClassifier(random_state=1)
    model.fit(X_test, y_test)
    preds = model.predict(X_test)
    
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F-beta should be between 0 and 1"

def test_inference(sample_data):
    X_train, y_train, X_test, _ = sample_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert preds.shape[0] == X_test.shape[0], "The number of predictions should match the number of test samples"
