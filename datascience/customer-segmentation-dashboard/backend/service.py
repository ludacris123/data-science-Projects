"""customer-segmentation-dashboard: project-specific business logic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

def segmentation(data):
    df=frame(data,['customer_id','date','amount'])
    df['date']=pd.to_datetime(df.date)
    if (df.amount < 0).any(): raise ValueError('Use positive purchases; handle refunds separately')
    snapshot=df.date.max()+pd.Timedelta(days=1)
    rfm=df.groupby('customer_id').agg(last=('date','max'),frequency=('date','count'),monetary=('amount','sum'))
    rfm['recency']=(snapshot-rfm.pop('last')).dt.days
    if len(rfm)<3: raise ValueError('At least three customers required')
    x=StandardScaler().fit_transform(np.log1p(rfm[['recency','frequency','monetary']]))
    algorithm=data.get('algorithm','kmeans')
    if algorithm=='dbscan': labels=DBSCAN(eps=float(data.get('eps',1)),min_samples=2).fit_predict(x)
    elif algorithm=='kmeans': labels=KMeans(n_clusters=min(int(data.get('clusters',3)),len(rfm)),random_state=42,n_init=10).fit_predict(x)
    else: raise ValueError('algorithm must be kmeans or dbscan')
    rfm['segment']=labels
    df['month']=df.date.dt.to_period('M')
    df['cohort']=df.groupby('customer_id').month.transform('min')
    df['period']=[m.ordinal-c.ordinal for m,c in zip(df.month,df.cohort)]
    counts=df.groupby(['cohort','period']).customer_id.nunique().unstack(fill_value=0)
    retention=counts.div(counts[0],axis=0)
    return {'customers':rfm.reset_index().to_dict('records'),'profiles':rfm.groupby('segment').mean(numeric_only=True).reset_index().to_dict('records'), 'retention':[{'cohort':str(c),'period':int(p),'retention':float(v)} for c,row in retention.iterrows() for p,v in row.items() if c.ordinal+p <= df.month.max().ordinal]}


run = segmentation
