"""Dense RAG with explicit, per-document-set namespaces."""
import hashlib
import json
import os
import httpx
import numpy as np


def embeddings(texts):
    from .ai import key
    token=key('OPENAI_API_KEY')
    with httpx.Client(timeout=90) as client:
        r=client.post('https://api.openai.com/v1/embeddings',headers={'Authorization':'Bearer '+token},json={'model':'text-embedding-3-small','input':texts});r.raise_for_status()
    return [x['embedding'] for x in sorted(r.json()['data'],key=lambda x:x['index'])]


def dense(documents,query,backend='chroma'):
    chunks=[]
    for doc in documents[:30]:
        text=str(doc.get('text',''))[:100000]
        for start in range(0,len(text),1200):chunks.append({'source':str(doc.get('name','document')),'text':text[start:start+1400]})
    if not chunks:raise ValueError('Supply nonempty documents')
    if len(chunks)>100:raise ValueError('Dense indexing limit: 100 chunks per request')
    namespace='docs-'+hashlib.sha256(json.dumps(chunks,sort_keys=True).encode()).hexdigest()[:24]
    vectors=embeddings([c['text'] for c in chunks]+[query])
    if backend=='chroma':
        import chromadb
        client=chromadb.PersistentClient(path=os.environ.get('CHROMA_DIR','chroma-data'))
        collection=client.get_or_create_collection(namespace,metadata={'hnsw:space':'cosine'})
        collection.upsert(ids=[str(i) for i in range(len(chunks))],embeddings=vectors[:-1],documents=[c['text'] for c in chunks],metadatas=[{'source':c['source']} for c in chunks])
        result=collection.query(query_embeddings=[vectors[-1]],n_results=min(5,len(chunks)))
        selected=[{'source':m['source'],'text':d,'score':1-float(s)} for m,d,s in zip(result['metadatas'][0],result['documents'][0],result['distances'][0])]
    elif backend=='pinecone':
        from pinecone import Pinecone
        from .ai import key
        client=Pinecone(api_key=key('PINECONE_API_KEY'))
        # Provision a 1536-dimensional cosine VECTOR index, not a document index.
        index=client.Index(host=key('PINECONE_INDEX_HOST'))
        index.upsert(vectors=[{'id':str(i),'values':v,'metadata':chunks[i]} for i,v in enumerate(vectors[:-1])],namespace=namespace)
        result=index.query(vector=vectors[-1],top_k=min(5,len(chunks)),include_metadata=True,namespace=namespace)
        selected=[{**m['metadata'],'score':float(m['score'])} for m in result['matches']]
    else:raise ValueError('retrieval must be tfidf, chroma or pinecone')
    return [{**item,'citation':f'[{i+1}]'} for i,item in enumerate(selected)]
