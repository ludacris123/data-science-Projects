import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from portfolio_core import analytics,ai,neural
from portfolio_core.app import create_app
ROOT=Path(__file__).resolve().parents[1]

def metadata(slug):return json.loads(next(ROOT.glob('*/'+slug+'/project.json')).read_text())

@pytest.mark.parametrize('slug',list(analytics.SERVICES))
def test_analytics_http(slug):
    meta=metadata(slug);client=TestClient(create_app(slug,meta));r=client.post('/run',json=meta['sample'])
    assert r.status_code==200,r.text
    assert isinstance(r.json(),dict) and r.json()
    assert client.get('/schema').json()['id']==slug
    assert client.post('/run',json={'rows':[]}).status_code==422


def test_forecast_chronological_intervals():
    output=analytics.outbreak(metadata('disease-outbreak-predictor')['sample'])['regions'][0]
    assert len(output['forecast'])==7
    assert output['forecast'][0]['date']=='2026-09-01'
    assert all(r['lower']<=r['prediction']<=r['upper'] for r in output['forecast'])
    assert output['backtest_mae']>=0


def test_rfm_totals_and_retention():
    data=metadata('customer-segmentation-dashboard')['sample'];r=analytics.segmentation(data)
    assert sum(c['frequency'] for c in r['customers'])==len(data['rows'])
    assert sum(c['monetary'] for c in r['customers'])==sum(c['amount'] for c in data['rows'])
    assert all(0<=c['retention']<=1 for c in r['retention'])
    assert all(c['retention']==1 for c in r['retention'] if c['period']==0)


def test_exports_and_csv():
    slug='personal-finance-platform';client=TestClient(create_app(slug));sample=metadata(slug)['sample']
    sample['rows'][0]['description']='=WEBSERVICE("bad")'
    assert "'=WEBSERVICE" in client.post('/export/csv',json=sample).text
    r=client.post('/export/pdf',json=sample);assert r.status_code==200 and r.content.startswith(b'%PDF')
    r=client.post('/upload-csv',files={'file':('bank.csv','date,description,amount\n2026-08-01,Salary,1000\n2026-08-02,Rent,-300\n','text/csv')})
    assert r.status_code==200 and r.json()['net_cashflow']==700


def test_retrieval_sources_no_match():
    assert ai.retrieve([{'name':'cats','text':'feline animals'}],'quantum physics')==[]
    sources=ai.retrieve([{'name':'doc','text':'FastAPI supports Python APIs'}],'Python FastAPI')
    assert sources[0]['source']=='doc' and sources[0]['citation']=='[1]'


def test_missing_keys_fail_explicitly(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    client=TestClient(create_app('ai-blog-writer'))
    assert client.post('/run',json=metadata('ai-blog-writer')['sample']).status_code==422


def test_langgraph_executes_bounded_plan():
    with patch.object(ai,'llm',side_effect=['["first", "second", "third", "fourth"]','Report']),patch.object(ai,'search',return_value=[{'url':'https://example.org','content':'Evidence','title':'source'}]) as search:
        r=ai.agent({'query':'topic'},'research')
    assert search.call_count==3
    assert len(r['sources'])==1 and r['report']=='Report'
    assert [e['stage'] for e in r['events']]==['plan','search','synthesize']


def test_bad_agent_plan():
    with patch.object(ai,'llm',return_value='{"unexpected":true}'):
        with pytest.raises(ValueError):ai.agent({'query':'topic'},'research')


def test_sign_validates_sequence_and_predicts():
    class FakeTensor:
        def numpy(self):return np.array([[.1,.9]])
    with patch.object(neural,'load_artifact',return_value=(lambda *a,**k:FakeTensor(),['A','B'])):
        assert neural.sign({'landmarks':np.zeros((30,126)).tolist()})['word']=='B'
        with pytest.raises(ValueError):neural.sign({'landmarks':[]})


def test_cors_only_configured_origin():
    client=TestClient(create_app('personal-finance-platform'))
    assert client.get('/health',headers={'Origin':'https://unknown.example'}).headers.get('access-control-allow-origin') is None


def test_body_limit(monkeypatch):
    import portfolio_core.app as runtime
    monkeypatch.setattr(runtime,'MAX_BODY',50)
    client=TestClient(create_app('personal-finance-platform'))
    assert client.post('/run',json={'rows':['x'*100]}).status_code==413
