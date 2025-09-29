import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import yaml
import mlflow
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse #used to get schema of remote repository

os.environ["MLFLOW_TRACKING_USERNAME"] = "somitrasinghkushwah"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "15a57c70d87fb0802f42b6e73bbb8c58b459abb2"

params = yaml.safe_load(open("params.yaml"))['train']






def train(data_path, model_path, random_state, n_estimatore, max_depth):
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1] 
    X = X.astype(float)  
    y = df.iloc[:, -1] 
    lr = LogisticRegression(max_iter=1000)
    param_grid = [
    {'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg', 'newton-cholesky'], 'fit_intercept': [True, False]},
    {'penalty': ['l1'], 'solver': ['liblinear'], 'fit_intercept': [True, False]}
]


    mlflow.set_tracking_uri('https://dagshub.com/somitrasinghkushwah/dagshub_practice.mlflow')

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        signature = infer_signature(X_train, y_train)

        grid = GridSearchCV(estimator=lr,param_grid=param_grid,cv=3, n_jobs=-1, verbose=True)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_param('penalty', grid.best_params_['penalty'])
        mlflow.log_param('fit_intercept', grid.best_params_['fit_intercept'])
        mlflow.log_param('solver', grid.best_params_['solver'])

        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)

        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!='file': ## if it http, like remote server
            mlflow.sklearn.log_model(best_model,"model",signature=signature, registered_model_name="Best Model")
        else: ## if its local filesystem
            mlflow.sklearn.log_model(best_model, "model",signature=signature)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_model, open(model_path, 'wb'))
        print(f"Model saved to {model_path}")





if __name__=="__main__":
    train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])



