import pickle
from sklearn.datasets import load_iris
from skelearn.model_selection import LogisticRegression

def train():
    # Load the dataset
    data = load_iris()

    X, y = data.data, data.target

    #Train model
    model = LogisticRegression(max_iter = 200)
    model.fit(X, y)

    # Save model
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and save as model.pkl")

    if __name__ == "__main__":
        train()
    