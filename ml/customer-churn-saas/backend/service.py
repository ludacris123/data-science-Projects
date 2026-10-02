"""customer-churn-saas: project-specific business logic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

def churn(data):
    if data.get("engine")=="advanced":
        from portfolio_core.advanced import churn as advanced_churn
        return advanced_churn(data)
    df=frame(data,['tenure','monthly_charges','support_tickets','churn'])
    if len(df)<40 or df.churn.nunique()!=2: raise ValueError('At least 40 records with both churn labels')
    cols=['tenure','monthly_charges','support_tickets']
    xtrain,xtest,ytrain,ytest=train_test_split(df[cols],df.churn,test_size=.25,stratify=df.churn,random_state=42)
    model=HistGradientBoostingClassifier(max_iter=80,random_state=42).fit(xtrain,ytrain)
    customers=pd.DataFrame(data.get('customers',df[cols].head(10).to_dict('records')))[cols]
    customers['risk']=model.predict_proba(customers)[:,1]
    customers['recommendation']=['Personal outreach and support review' if r>=.6 else 'Monitor engagement' if r>=.3 else 'Standard engagement' for r in customers.risk]
    return {'roc_auc':float(roc_auc_score(ytest,model.predict_proba(xtest)[:,1])),'customers':customers.to_dict('records')}


run = churn
