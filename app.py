import os
import warnings
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = root_mean_squared_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mae, mse, rmse, r2

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(40)

    csv_url = 'https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv'  # Corrected URL
    try:
        df = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception('Unable to download training and testing CSV: %s', e) #added exception print
        sys.exit(1) #exit the program.

    train, test = train_test_split(df)
    train_x = train.drop(columns=["quality"], axis=1) #corrected column drop
    test_x = test.drop(columns=["quality"], axis=1) #corrected column drop
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        test_x_pred = lr.predict(test_x)
        (mae, mse, rmse, r2) = eval_metrics(test_y, test_x_pred)
        print(f"{alpha}->alpha and {l1_ratio}->l1 ratio")
        print(f"Mae->{mae}, Mse->{mse}, Rmse->{rmse}, R2->{r2}")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        train_x_pred = lr.predict(train_x)
        signature = infer_signature(train_x, train_x_pred)

        tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_score != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticNetwineModel", signature=signature)
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)