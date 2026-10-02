import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

def house(data):
    from xgboost import XGBRegressor
    import shap
    import mlflow
    df=frame(data,['bedrooms','area','age','price']);cols=['bedrooms','area','age']
    if len(df)<40:raise ValueError('At least 40 houses required')
    train,test=train_test_split(df,test_size=.2,random_state=42)
    model=XGBRegressor(n_estimators=200,max_depth=3,learning_rate=.05,random_state=42).fit(train[cols],train.price)
    query=pd.DataFrame([data['house']])[cols]
    explainer=shap.TreeExplainer(model);values=explainer(query)
    mae=float(mean_absolute_error(test.price,model.predict(test[cols])))
    mlflow.set_experiment('real-estate-portfolio')
    with mlflow.start_run() as run:
        mlflow.log_params({'n_estimators':200,'max_depth':3});mlflow.log_metric('holdout_mae',mae)
        mlflow.xgboost.log_model(model,artifact_path='model')
        run_id=run.info.run_id
    return {'prediction':float(model.predict(query)[0]),'test_mae':mae,'shap_values':dict(zip(cols,map(float,values.values[0]))),'shap_base_value':float(np.asarray(values.base_values).ravel()[0]),'mlflow_run_id':run_id}


run = house
