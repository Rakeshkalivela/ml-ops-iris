# Script for pipelines quality control. 
#  If model accuracy dropps below benchmark, 
# the pipeline will fain and stop deployement.

import os
import pickle
from sklearn.datasets import load_iris

def test_model_accuracy():
    # ensure modele file exists
    assert os.path.exists("model.pkl"), "Model file was not created"

    # Load model and data
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    data = load_iris()

    score = model.score(data.data, data.target)
    print(f"Current pipeline model accuracy: {score:.2f}")

    # Quality gate threshold 90%
    assert score >= 0.9, f"Model accuracy {score:.2f} dropped below minimum threshold 0.90"

