from sklearn.linear_model import LogisticRegression
from src.utils import read_yaml
import joblib


def training(dataset_path: str, model_path: str):
    iris = joblib.load(dataset_path)

    log_reg = LogisticRegression(max_iter=7)
    log_reg.fit(iris["X_train"], iris["y_train"])

    joblib.dump(log_reg, model_path)


if __name__ == "__main__":
    dataset_path = read_yaml("params.yaml")["training"]["dataset_path"]
    model_path = read_yaml("params.yaml")["training"]["model_path"]

    training(dataset_path, model_path)
