from sklearn import datasets
import joblib
from src.utils import read_yaml
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def preprocessing(out_path: str):
    iris = datasets.load_iris()

    # iris = joblib.load(dataset_path)
    iris = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )

    # Droping the target and species since we only need the measurements
    X = iris.drop(['target'], axis=1)

    # converting into numpy array and assigning petal length and petal width
    X = X.to_numpy()[:, (2, 3)]
    y = iris['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        },
        out_path
    )


if __name__ == "__main__":
    out_path = read_yaml("params.yaml")["preprocessing"]["dataset_path"]
    preprocessing(out_path)
