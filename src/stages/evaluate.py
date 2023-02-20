from sklearn import metrics
from src.utils import read_yaml
import joblib
import json


def evaluate(dataset_path: str, model_path: str, metrics_path: str):
    iris = joblib.load(dataset_path)
    model = joblib.load(model_path)
    preds = model.predict(iris["X_test"])
    f1_score = metrics.f1_score(iris["y_test"], preds, average="macro")
    with open(metrics_path, 'w') as fp:
        json.dump({'f1_score': f1_score}, fp)


if __name__ == "__main__":
    dataset_path = read_yaml("params.yaml")["evaluate"]["dataset_path"]
    model_path = read_yaml("params.yaml")["evaluate"]["model_path"]
    metrics_path = read_yaml("params.yaml")["evaluate"]["metrics_path"]

    evaluate(dataset_path, model_path, metrics_path)
