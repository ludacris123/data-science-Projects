"""interior-design-studio: project-specific business logic."""
import base64
import json
import httpx
import os
from portfolio_core import ai

def interior(data):
    token=ai.key('REPLICATE_API_TOKEN'); version=ai.key('REPLICATE_MODEL_VERSION')
    image=data.get('image_base64','')
    raw=base64.b64decode(image,validate=True)
    if not raw or len(raw)>10_000_000: raise ValueError('Supply an image up to 10 MB')
    style=data.get('style','modern')
    if style not in ['modern','boho','minimal']: raise ValueError('Unknown style')
    # Select a ControlNet-compatible model version and its exact field names.
    inputs={os.environ.get('REPLICATE_IMAGE_FIELD','image'):'data:image/png;base64,'+image,'prompt':f'{style} interior design, preserve room layout'}
    with httpx.Client(timeout=90) as client:
        r=client.post('https://api.replicate.com/v1/predictions',headers={'Authorization':'Bearer '+token},json={'version':version,'input':inputs}); r.raise_for_status()
    body=r.json()
    return {'prediction_id':body['id'],'status':body['status'],'output':body.get('output'),'poll_path':'/provider/replicate/'+body['id']}


run = interior
