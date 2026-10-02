import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

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


run = churn
