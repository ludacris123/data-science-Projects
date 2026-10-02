import csv
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from . import analytics, neural, ai
from .registry import load_service, metadata as project_metadata

SERVICES={**analytics.SERVICES,**neural.SERVICES,**ai.SERVICES}
MAX_BODY=30_000_000


def create_app(project,metadata=None):
    if project not in SERVICES: raise ValueError('Unknown project')
    metadata=metadata or project_metadata(project)
    app=FastAPI(title=project,version='0.2.0')
    app.add_middleware(CORSMiddleware,allow_origins=os.environ.get('CORS_ORIGINS','http://localhost:5173').split(','),allow_methods=['GET','POST'],allow_headers=['Content-Type'])
    @app.middleware('http')
    async def limit_body(request,call_next):
        from starlette.responses import JSONResponse
        # Read and cap streaming bodies as well as requests with Content-Length.
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>MAX_BODY: return JSONResponse({'detail':'Request exceeds 30 MB'},status_code=413)
        request._body=bytes(body)
        return await call_next(request)
    @app.get('/health')
    def health(): return {'status':'ok','project':project}
    @app.get('/schema')
    def schema(): return metadata or {'project':project}
    async def execute(data):
        try:
            from . import persistence
            if project=='stock-sentiment-analyzer':
                existing=await run_in_threadpool(persistence.cached,project,data)
                if existing:return existing
            result=await run_in_threadpool(load_service(project).run,data)
            if os.environ.get('DATABASE_URL'):await run_in_threadpool(persistence.save,project,result)
            if project=='stock-sentiment-analyzer':await run_in_threadpool(persistence.cache,project,data,result)
            return result
        except (ValueError,KeyError,TypeError) as e: raise HTTPException(422,str(e)) from e
        except ImportError as e: raise HTTPException(503,'Optional dependency missing: '+str(e)) from e
        except httpx.HTTPError as e: raise HTTPException(502,'External provider request failed') from e
    @app.get('/history')
    def history():
        from .persistence import history
        try:return history(project)
        except ValueError as e:raise HTTPException(503,str(e)) from e
    @app.post('/run')
    async def run(data:dict): return await execute(data)
    @app.post('/upload-csv')
    async def upload(file:UploadFile=File(...)):
        if project not in analytics.SERVICES: raise HTTPException(422,'This project does not consume CSV')
        import pandas as pd
        try: rows=pd.read_csv(io.BytesIO(await file.read())).to_dict('records')
        except Exception as e: raise HTTPException(422,'Invalid UTF-8 CSV') from e
        return await execute({'rows':rows})
    @app.post('/export/csv')
    async def export(data:dict):
        result=await execute(data)
        rows=next((v for v in result.values() if isinstance(v,list) and v and isinstance(v[0],dict)),[])
        out=io.StringIO()
        if rows:
            writer=csv.DictWriter(out,fieldnames=rows[0].keys());writer.writeheader()
            for row in rows:
                writer.writerow({k: "'"+str(v) if isinstance(v,str) and v.startswith(('=','+','-','@')) else v for k,v in row.items()})
        return StreamingResponse(iter([out.getvalue()]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=results.csv'})
    @app.post('/export/pdf')
    async def pdf(data:dict):
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import simpleSplit
        result=await execute(data);out=io.BytesIO();c=canvas.Canvas(out);y=800
        for line in simpleSplit(json.dumps(result,indent=2,ensure_ascii=True),'Helvetica',9,510):
            c.drawString(40,y,line);y-=12
            if y<40:c.showPage();y=800
        c.save();out.seek(0)
        return StreamingResponse(out,media_type='application/pdf',headers={'Content-Disposition':'attachment; filename=report.pdf'})
    @app.post('/stream')
    async def stream(data:dict):
        # Stream real node updates, never private model reasoning.
        if project not in ['research-assistant','job-application-agent','shopping-agent']: raise HTTPException(422,'Streaming is available for agent projects')
        kind={'research-assistant':'research','job-application-agent':'jobs','shopping-agent':'shopping'}[project]
        def events():
            try:
                query=str(data.get('query',''))
                if kind=='jobs':query+='\nResume: '+str(data.get('resume',''))[:20000]
                for update in load_service(project).stream(data):
                    yield 'data: '+json.dumps(update)+'\n\n'
            except Exception:
                yield 'event: error\ndata: '+json.dumps({'detail':'Agent failed; check provider configuration and logs'})+'\n\n'
        return StreamingResponse(events(),media_type='text/event-stream')
    @app.get('/provider/replicate/{prediction_id}')
    def poll(prediction_id:str):
        if project!='interior-design-studio' or not re.fullmatch('[A-Za-z0-9_-]{1,100}',prediction_id):raise HTTPException(422,'Invalid prediction ID')
        try:
            r=httpx.get('https://api.replicate.com/v1/predictions/'+prediction_id,headers={'Authorization':'Bearer '+ai.key('REPLICATE_API_TOKEN')},timeout=30);r.raise_for_status();body=r.json()
            return {k:body.get(k) for k in ['id','status','output','error']}
        except ValueError as e:raise HTTPException(503,str(e)) from e
        except httpx.HTTPError as e:raise HTTPException(502,'Provider poll failed') from e
    return app
