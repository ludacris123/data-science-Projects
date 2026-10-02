import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from portfolio_core import advanced,retrieval,neural
ROOT=Path(__file__).resolve().parents[1]
def sample(slug):return json.loads(next(ROOT.glob('*/'+slug+'/project.json')).read_text())['sample']


def test_xgboost_shap_mlflow(tmp_path,monkeypatch):
    pytest.importorskip('xgboost');pytest.importorskip('shap')
    mlflow=pytest.importorskip('mlflow');mlflow.set_tracking_uri('sqlite:///'+str(tmp_path/'mlflow.sqlite3'))
    result=advanced.house(sample('real-estate-predictor'))
    assert result['test_mae']>=0
    assert abs(result['shap_base_value']+sum(result['shap_values'].values())-result['prediction'])<.2
    assert len(result['mlflow_run_id'])==32


def test_lightgbm_optuna():
    pytest.importorskip('lightgbm');pytest.importorskip('optuna')
    data=sample('customer-churn-saas');data['trials']=2
    result=advanced.churn(data)
    assert 0<=result['roc_auc']<=1 and 0<=result['cv_auc']<=1
    assert all(0<=c['risk']<=1 for c in result['customers'])


def test_chroma_persistence_with_supplied_embeddings(tmp_path,monkeypatch):
    pytest.importorskip('chromadb');monkeypatch.setenv('CHROMA_DIR',str(tmp_path))
    vectors=[[1.,0.,0.],[0.,1.,0.],[1.,0.,0.]]
    with patch.object(retrieval,'embeddings',return_value=vectors):
        result=retrieval.dense([{'name':'a','text':'Python APIs'},{'name':'b','text':'Gardening'}],'Python','chroma')
    assert result[0]['source']=='a' and result[0]['score']>.99


def test_faiss_similarity_pipeline():
    pytest.importorskip('faiss')
    class Tensor:
        def __init__(self,v):self.v=v
        def numpy(self):return np.array(self.v)
    class Encoder:
        def __call__(self,images,**kw):
            return Tensor([[1.,0.],[0.,1.]] if len(images)==2 else [[1.,0.]])
    with patch.object(neural,'image_array',return_value=np.zeros((1,224,224,3))),patch.object(neural,'image_encoder',return_value=Encoder()):
        r=neural.visual({'image_base64':'x','catalog':[{'id':'a','image_base64':'x'},{'id':'b','image_base64':'y'}]})
    assert r['matches'][0]['id']=='a' and r['matches'][0]['similarity']>.99
