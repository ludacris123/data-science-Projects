"""Provider-backed GenAI and bounded LangGraph workflows."""
import base64
import json
import os
import sqlite3
from pathlib import Path
from typing import TypedDict
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def key(name):
    value=os.environ.get(name)
    if not value: raise ValueError(name+' is required for this integration')
    return value


def llm(system,prompt):
    # Fixed provider URL; no caller-controlled network destinations.
    token=key('OPENAI_API_KEY')
    with httpx.Client(timeout=90) as client:
        response=client.post('https://api.openai.com/v1/chat/completions',headers={'Authorization':'Bearer '+token},json={'model':os.environ.get('OPENAI_MODEL','gpt-4o-mini'),'messages':[{'role':'system','content':system},{'role':'user','content':prompt}], 'temperature':.2})
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']


def retrieve(documents,query):
    chunks=[]
    for doc in documents[:30]:
        text=str(doc.get('text',''))[:100000]
        for start in range(0,len(text),1200):
            chunks.append({'source':str(doc.get('name','document')),'text':text[start:start+1400]})
    if not chunks: raise ValueError('Provide at least one nonempty document')
    vectorizer=TfidfVectorizer(stop_words='english')
    vectors=vectorizer.fit_transform([c['text'] for c in chunks]+[query])
    scores=cosine_similarity(vectors[-1],vectors[:-1])[0]
    ids=scores.argsort()[::-1][:5]
    return [{**chunks[int(i)],'score':float(scores[i]),'citation':f'[{j+1}]'} for j,i in enumerate(ids) if scores[i]>0]


def blog(data):
    from .registry import load_service
    return load_service('ai-blog-writer').run(data)


def interior(data):
    from .registry import load_service
    return load_service('interior-design-studio').run(data)


def transcribe(encoded):
    raw=base64.b64decode(encoded,validate=True)
    if not raw or len(raw)>20_000_000: raise ValueError('Audio must be 1 byte–20 MB')
    with httpx.Client(timeout=120) as client:
        r=client.post('https://api.openai.com/v1/audio/transcriptions',headers={'Authorization':'Bearer '+key('OPENAI_API_KEY')},data={'model':'whisper-1','response_format':'verbose_json'},files={'file':('episode.mp3',raw,'audio/mpeg')}); r.raise_for_status()
    return r.json()


def podcast(data):
    from .registry import load_service
    return load_service('podcast-summarizer').run(data)


def search(query):
    with httpx.Client(timeout=30) as client:
        r=client.post('https://api.tavily.com/search',json={'api_key':key('TAVILY_API_KEY'),'query':query,'max_results':5});r.raise_for_status()
    return [{'title':x['title'],'url':x['url'],'content':x.get('content','')[:4000]} for x in r.json().get('results',[])]


class State(TypedDict,total=False):
    query:str
    plan:list[str]
    sources:list[dict]
    report:str


def graph(kind):
    from .registry import load_service
    project={'research':'research-assistant','jobs':'job-application-agent','shopping':'shopping-agent'}.get(kind)
    if not project:raise ValueError('Unknown agent kind')
    return load_service(project).build_graph()


def agent(data,kind):
    query=data.get('query','')
    if kind=='jobs': query+='\nResume: '+str(data.get('resume',''))[:20000]
    if not query.strip(): raise ValueError('query is required')
    if kind=='shopping' and data.get('product_urls'):
        from .sources import browse_products
        pages=browse_products(query,data['product_urls'])
        return {'sources':pages,'report':llm('Compare these products against the request. Cite URLs, use only observed prices, never purchase.',json.dumps({'query':query,'pages':pages}))}
    updates=[]; final={}
    for update in graph(kind).stream({'query':query},stream_mode='updates'):
        for stage,values in update.items():
            final.update(values);updates.append({'stage':stage,'data':values})
    return {**final,'events':updates,'mode':'search and draft; no external applications or purchases'}

SERVICES={'ai-blog-writer':blog,'interior-design-studio':interior,'podcast-summarizer':podcast,'research-assistant':lambda d:agent(d,'research'),'job-application-agent':lambda d:agent(d,'jobs'),'shopping-agent':lambda d:agent(d,'shopping')}
