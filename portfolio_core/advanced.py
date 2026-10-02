"""Optional advanced ML engines; called when engine='advanced'."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error,roc_auc_score
from .analytics import frame


def house(data):
    from .registry import load_component
    return load_component('real-estate-predictor', 'advanced').run(data)


def churn(data):
    from .registry import load_component
    return load_component('customer-churn-saas', 'advanced').run(data)


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
