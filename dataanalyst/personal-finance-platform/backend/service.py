"""personal-finance-platform: project-specific business logic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from portfolio_core.analytics import frame

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


run = finance
