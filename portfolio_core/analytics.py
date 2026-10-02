"""Statistical services. Each function uses user data, not canned predictions."""
import io
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score


def frame(data, columns):
    if data.get('source'):
        from .sources import dataset
        rows=dataset(data['source'])
    else:rows=data.get('rows',[])
    df = pd.DataFrame(rows)
    if not set(columns).issubset(df.columns):
        raise ValueError('Required columns: ' + ', '.join(columns))
    if df.empty or len(df) > 50000:
        raise ValueError('Supply 1–50,000 records')
    return df


def outbreak(data):
    from .registry import load_service
    return load_service('disease-outbreak-predictor').run(data)


def segmentation(data):
    from .registry import load_service
    return load_service('customer-segmentation-dashboard').run(data)


def sentiment(data):
    from .registry import load_service
    return load_service('stock-sentiment-analyzer').run(data)


def finance(data):
    from .registry import load_service
    return load_service('personal-finance-platform').run(data)


def streaming(data):
    from .registry import load_service
    return load_service('streaming-content-insights').run(data)


def sports(data):
    from .registry import load_service
    return load_service('sports-analytics-hub').run(data)


def real_estate(data):
    from .registry import load_service
    return load_service('real-estate-predictor').run(data)


def churn(data):
    from .registry import load_service
    return load_service('customer-churn-saas').run(data)

SERVICES={'disease-outbreak-predictor':outbreak,'customer-segmentation-dashboard':segmentation,'stock-sentiment-analyzer':sentiment,'personal-finance-platform':finance,'streaming-content-insights':streaming,'sports-analytics-hub':sports,'real-estate-predictor':real_estate,'customer-churn-saas':churn}
