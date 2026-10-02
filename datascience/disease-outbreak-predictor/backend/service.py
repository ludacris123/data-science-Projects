"""disease-outbreak-predictor: project-specific business logic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

def outbreak(data):
    from statsmodels.tsa.arima.model import ARIMA
    df = frame(data, ['date', 'region', 'cases'])
    df['date'] = pd.to_datetime(df.date)
    if (df.cases < 0).any(): raise ValueError('Cases must be nonnegative')
    horizon = int(data.get('horizon', 7))
    if not 1 <= horizon <= 60: raise ValueError('Horizon must be 1–60')
    output = []
    for region, group in df.groupby('region'):
        ts = group.groupby('date').cases.sum().asfreq('D')
        if ts.isna().any(): raise ValueError('Daily series has gaps; fill explicitly before forecasting')
        if len(ts) < 20: raise ValueError('At least 20 daily observations per region')
        holdout = min(7, len(ts)//5)
        backtest = ARIMA(ts.iloc[:-holdout], order=(1,1,0)).fit().forecast(holdout)
        model = ARIMA(ts, order=(1,1,0)).fit()
        forecast = model.get_forecast(horizon)
        bounds = forecast.conf_int(alpha=.05)
        output.append({'region': str(region), 'backtest_mae':float(mean_absolute_error(ts.iloc[-holdout:], backtest)), 'forecast':[{'date':str(t.date()),'prediction':float(max(0,v)), 'lower':float(max(0,bounds.iloc[i,0])), 'upper':float(max(0,bounds.iloc[i,1]))} for i,(t,v) in enumerate(forecast.predicted_mean.items())]})
    return {'regions':output,'interval':'95% model-based interval; not calibrated for policy use'}


run = outbreak
