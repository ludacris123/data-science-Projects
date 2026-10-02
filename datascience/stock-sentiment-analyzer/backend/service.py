"""stock-sentiment-analyzer: project-specific business logic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

def sentiment(data):
    if data.get("symbol"):
        from portfolio_core.advanced import market
        return market(data)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    df=frame(data,['date','headline','close'])
    analyzer=SentimentIntensityAnalyzer()
    df['sentiment']=[analyzer.polarity_scores(str(s))['compound'] for s in df.headline]
    daily=df.groupby('date').agg(close=('close','last'),sentiment=('sentiment','mean')).sort_index()
    daily['return']=daily.close.pct_change()
    paired=daily[['sentiment','return']].dropna()
    corr=paired.corr().iloc[0,1] if len(paired)>2 else np.nan
    return {'headlines':df.to_dict('records'),'daily':daily.reset_index().replace({np.nan:None}).to_dict('records'),'correlation':float(corr) if np.isfinite(corr) else None,'method':'VADER baseline; correlation is contemporaneous, not causal'}


run = sentiment
