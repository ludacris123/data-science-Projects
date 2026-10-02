"""streaming-content-insights: project-specific business logic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

def streaming(data):
    import duckdb
    df=frame(data,['title','genre','runtime','rating','year'])
    if data.get('genre_filter'):
        df=df[df.genre.str.casefold()==str(data['genre_filter']).casefold()]
        if df.empty:raise ValueError('No titles match this genre')
    connection=duckdb.connect(':memory:')
    connection.register('content',df)
    try:
        groups=connection.execute('SELECT genre, count(*) AS titles, avg(rating) AS rating, avg(runtime) AS runtime FROM content GROUP BY genre ORDER BY rating DESC').df().to_dict('records')
    finally: connection.close()
    corr=df[['runtime','rating']].corr().iloc[0,1]
    return {'genres':groups,'runtime_rating_correlation':float(corr) if np.isfinite(corr) else None,'titles':df.sort_values('rating',ascending=False).to_dict('records')}


run = streaming
