"""sports-analytics-hub: project-specific business logic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

def sports(data):
    df=frame(data,['player','runs','balls','wickets','matches'])
    if (df[['balls','matches']]<=0).any().any(): raise ValueError('Balls and matches must be positive')
    df['strike_rate']=df.runs/df.balls*100
    df['runs_per_match']=df.runs/df.matches
    df['wickets_per_match']=df.wickets/df.matches
    metrics=df[['strike_rate','runs_per_match','wickets_per_match']]
    radar=(metrics-metrics.min())/(metrics.max()-metrics.min()).replace(0,1)
    radar['player']=df.player
    result={'players':df.to_dict('records'),'radar':radar.to_dict('records')}
    if data.get('matches'):
        from sklearn.linear_model import LogisticRegression
        matches=pd.DataFrame(data['matches']);cols=['runs_needed','balls_remaining','wickets_remaining']
        if not set(cols+['won']).issubset(matches.columns) or len(matches)<60:raise ValueError('Provide 60+ match states with runs_needed, balls_remaining, wickets_remaining, won')
        if matches.won.value_counts().min()<10:raise ValueError('At least 10 examples per outcome')
        train,test=train_test_split(matches,test_size=.25,stratify=matches.won,random_state=42)
        model=make_pipeline(StandardScaler(),LogisticRegression()).fit(train[cols],train.won)
        result['win_probability']=float(model.predict_proba(pd.DataFrame([data['match_state']])[cols])[0,1])
        result['win_model_auc']=float(roc_auc_score(test.won,model.predict_proba(test[cols])[:,1]))
        result['evaluation_note']='Split by match before creating state rows to avoid leakage; this row-level demo is not a validated live model'
    return result


run = sports
