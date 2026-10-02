"""Opt-in PostgreSQL runs and Redis cache. Configure only for a trusted local workspace."""
import hashlib
import json
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def engine():
    from sqlalchemy import create_engine, text
    url=os.environ.get('DATABASE_URL')
    if not url: return None
    e=create_engine(url,pool_pre_ping=True)
    with e.begin() as c:c.execute(text('CREATE TABLE IF NOT EXISTS portfolio_runs (id BIGSERIAL PRIMARY KEY, project TEXT NOT NULL, result JSONB NOT NULL, created_at TIMESTAMPTZ DEFAULT now())'))
    return e

def save(project,result):
    from sqlalchemy import text
    e=engine()
    if e is None:return
    with e.begin() as c:c.execute(text('INSERT INTO portfolio_runs(project,result) VALUES (:project,CAST(:result AS JSONB))'),{'project':project,'result':json.dumps(result)})

def history(project):
    from sqlalchemy import text
    e=engine()
    if e is None:raise ValueError('DATABASE_URL required for persistent history')
    with e.connect() as c:return [dict(r._mapping) for r in c.execute(text('SELECT id,project,result,created_at FROM portfolio_runs WHERE project=:project ORDER BY id DESC LIMIT 20'),{'project':project})]

def cache_key(project,data):return 'portfolio:'+project+':'+hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

def cached(project,data):
    if not os.environ.get('REDIS_URL'):return None
    from redis import Redis
    c=Redis.from_url(os.environ['REDIS_URL'],socket_timeout=3)
    value=c.get(cache_key(project,data));return json.loads(value) if value else None

def cache(project,data,result):
    if not os.environ.get('REDIS_URL'):return
    from redis import Redis
    c=Redis.from_url(os.environ['REDIS_URL'],socket_timeout=3)
    c.setex(cache_key(project,data),300,json.dumps(result))
