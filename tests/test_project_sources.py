import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from portfolio_core.app import create_app
from portfolio_core.registry import load_service,project_path
from portfolio_core import ai

ROOT=Path(__file__).resolve().parents[1]
CATALOG=json.loads((ROOT/'catalog.json').read_text())

@pytest.mark.parametrize('project',[p['id'] for p in CATALOG])
def test_project_code_and_metadata(project):
    path=project_path(project)
    assert callable(load_service(project).run)
    assert (path/'frontend/src/main.jsx').is_file()
    assert (path/'frontend/package.json').is_file()
    assert (path/'backend/run.py').is_file()
    client=TestClient(create_app(project))
    meta=client.get('/schema').json()
    assert meta['id']==project and meta['sample']


def test_streaming_genre_filter():
    sample=json.loads((project_path('streaming-content-insights')/'sample-request.json').read_text())
    sample['genre_filter']='Drama'
    result=load_service('streaming-content-insights').run(sample)
    assert all(row['genre']=='Drama' for row in result['titles'])
    sample['genre_filter']='nonexistent'
    with pytest.raises(ValueError):load_service('streaming-content-insights').run(sample)


def test_agent_owned_stream(monkeypatch):
    def llm(system,prompt):
        return '["test search"]' if 'JSON array' in system else 'Cited report'
    monkeypatch.setattr(ai,'llm',llm)
    monkeypatch.setattr(ai,'search',lambda query:[{'title':'Evidence','url':'https://example.org','content':'Source text'}])
    service=load_service('research-assistant')
    assert service.run({'query':'test'})['report']=='Cited report'
    events=list(service.stream({'query':'test'}))
    assert [next(iter(event)) for event in events]==['plan','search','synthesize']
    with pytest.raises(ValueError):service.stream({'query':''})
