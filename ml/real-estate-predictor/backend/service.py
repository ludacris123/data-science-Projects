"""real-estate-predictor: project-specific business logic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

def real_estate(data):
    if data.get("engine")=="advanced":
        from portfolio_core.advanced import house
        return house(data)
    df=frame(data,['bedrooms','area','age','price'])
    if len(df)<30: raise ValueError('At least 30 houses required')
    cols=['bedrooms','area','age']
    xtrain,xtest,ytrain,ytest=train_test_split(df[cols],df.price,test_size=.2,random_state=42)
    model=RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=42).fit(xtrain,ytrain)
    query=pd.DataFrame([data.get('house',{'bedrooms':3,'area':1500,'age':5})])[cols]
    prediction=float(model.predict(query)[0])
    baseline=model.predict(pd.DataFrame([xtrain.median()]))[0]
    changes=[]
    for col in cols:
        altered=query.copy(); altered[col]=xtrain[col].median()
        changes.append({'feature':col,'prediction_difference':float(prediction-model.predict(altered)[0])})
    return {'prediction':prediction,'test_mae':float(mean_absolute_error(ytest,model.predict(xtest))),'median_house_prediction':float(baseline),'feature_sensitivity':changes,'method':'Random forest baseline; sensitivity is not SHAP'}


run = real_estate
