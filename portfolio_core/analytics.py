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


def sentiment(data):
    if data.get("symbol"):
        from .advanced import market
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


def finance(data):
    df=frame(data,['date','description','amount'])
    df['date']=pd.to_datetime(df.date)
    rules={'groceries':['grocery','supermarket'],'transport':['uber','fuel','taxi'],'rent':['rent'],'dining':['restaurant','cafe','zomato'],'income':['salary','payroll']}
    df['category']=[next((k for k,words in rules.items() if any(w in str(s).lower() for w in words)),'other') for s in df.description]
    df['month']=df.date.dt.strftime('%Y-%m')
    df['expense']=(-df.amount).clip(lower=0)
    summary=df.groupby(['month','category']).expense.sum().reset_index()
    budgets=data.get('budgets',{})
    summary['budget']=[float(budgets.get(c,0)) for c in summary.category]
    summary['remaining']=summary.budget-summary.expense
    df['date']=df.date.dt.strftime('%Y-%m-%d')
    return {'transactions':df.to_dict('records'),'budget_vs_actual':summary.to_dict('records'),'net_cashflow':float(df.amount.sum())}


def streaming(data):
    import duckdb
    df=frame(data,['title','genre','runtime','rating','year'])
    connection=duckdb.connect(':memory:')
    connection.register('content',df)
    try:
        groups=connection.execute('SELECT genre, count(*) AS titles, avg(rating) AS rating, avg(runtime) AS runtime FROM content GROUP BY genre ORDER BY rating DESC').df().to_dict('records')
    finally: connection.close()
    corr=df[['runtime','rating']].corr().iloc[0,1]
    return {'genres':groups,'runtime_rating_correlation':float(corr) if np.isfinite(corr) else None,'titles':df.sort_values('rating',ascending=False).to_dict('records')}


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


def real_estate(data):
    if data.get("engine")=="advanced":
        from .advanced import house
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


def churn(data):
    if data.get("engine")=="advanced":
        from .advanced import churn as advanced_churn
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

SERVICES={'disease-outbreak-predictor':outbreak,'customer-segmentation-dashboard':segmentation,'stock-sentiment-analyzer':sentiment,'personal-finance-platform':finance,'streaming-content-insights':streaming,'sports-analytics-hub':sports,'real-estate-predictor':real_estate,'customer-churn-saas':churn}
