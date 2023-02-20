from sklearn.linear_model import LogisticRegression
from src.utils import read_yaml
import joblib
from typing import Dict


def training(dataset_path: str, model_path: str, model_kw: Dict):
    iris = joblib.load(dataset_path)

    log_reg = LogisticRegression(**model_kw)
    log_reg.fit(iris["X_train"], iris["y_train"])

    joblib.dump(log_reg, model_path)


if __name__ == "__main__":
    dataset_path = read_yaml("params.yaml")["training"]["dataset_path"]
    model_path = read_yaml("params.yaml")["training"]["model_path"]
    model_kw = read_yaml("params.yaml")["training"]["model_kw"]

    training(dataset_path, model_path, model_kw)
