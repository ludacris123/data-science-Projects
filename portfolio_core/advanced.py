"""Optional advanced ML engines; called when engine='advanced'."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error,roc_auc_score
from .analytics import frame


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


def churn(data):
    from lightgbm import LGBMClassifier
    import optuna
    df=frame(data,['tenure','monthly_charges','support_tickets','churn']);cols=['tenure','monthly_charges','support_tickets']
    if len(df)<60 or df.churn.value_counts().min()<12: raise ValueError('Supply at least 60 records with at least 12 of each class')
    train,test=train_test_split(df,test_size=.25,stratify=df.churn,random_state=42)
    folds=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    def objective(trial):
        model=LGBMClassifier(n_estimators=trial.suggest_int('n_estimators',50,200),num_leaves=trial.suggest_int('num_leaves',4,24),learning_rate=trial.suggest_float('learning_rate',.02,.2,log=True),min_child_samples=5,random_state=42,verbosity=-1,n_jobs=1)
        return float(cross_val_score(model,train[cols],train.churn,cv=folds,scoring='roc_auc').mean())
    trials=int(data.get('trials',10))
    if not 1<=trials<=30:raise ValueError('trials must be 1–30')
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42));study.optimize(objective,n_trials=trials)
    model=LGBMClassifier(**study.best_params,min_child_samples=5,random_state=42,verbosity=-1,n_jobs=1).fit(train[cols],train.churn)
    customers=pd.DataFrame(data.get('customers',df[cols].head(10).to_dict('records')))[cols]
    customers['risk']=model.predict_proba(customers)[:,1]
    customers['recommendation']=['Review support and offer retention outreach' if p>.6 else 'Monitor engagement' for p in customers.risk]
    return {'roc_auc':float(roc_auc_score(test.churn,model.predict_proba(test[cols])[:,1])),'cv_auc':study.best_value,'best_parameters':study.best_params,'customers':customers.to_dict('records')}


def market(data):
    import yfinance as yf
    from .analytics import sentiment
    symbol=str(data.get('symbol','AAPL'))
    if not symbol.isalnum() or len(symbol)>10:raise ValueError('Invalid ticker')
    prices=yf.Ticker(symbol).history(period='3mo')
    headlines=data.get('headlines',[])
    if not headlines:raise ValueError('Supply licensed headlines with date and headline')
    rows=[]
    for h in headlines:
        date=str(h['date'])[:10]
        match=prices[prices.index.strftime('%Y-%m-%d')==date]
        if len(match):rows.append({'date':date,'headline':h['headline'],'close':float(match.Close.iloc[0]),'open':float(match.Open.iloc[0]),'high':float(match.High.iloc[0]),'low':float(match.Low.iloc[0])})
    return sentiment({'rows':rows})
