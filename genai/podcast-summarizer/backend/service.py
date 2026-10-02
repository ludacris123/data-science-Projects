"""podcast-summarizer: project-specific business logic."""
import base64
import json
import httpx
import os
from portfolio_core import ai

def podcast(data):
    transcript=data.get('transcript')
    segments=[]
    if not transcript:
        result=ai.transcribe(data['audio_base64']); transcript=result['text']; segments=result.get('segments',[])
    if len(transcript)>200000: raise ValueError('Transcript exceeds 200,000 characters; split the episode')
    if data.get('question'):
        if data.get('retrieval')=='chroma':
            from portfolio_core.retrieval import dense
            sources=dense([{'name':'episode','text':transcript}],data['question'],'chroma')
        else:sources=ai.retrieve([{'name':'episode','text':transcript}],data['question'])
        answer=ai.llm('Answer only from transcript excerpts, citing [n]. If unsupported say so.',json.dumps({'question':data['question'],'sources':sources}))
        return {'transcript':transcript,'answer':answer,'sources':sources}
    excerpts=[transcript[i:i+12000] for i in range(0,len(transcript),12000)]
    notes=[ai.llm('Summarize this transcript chunk faithfully. Include key topics and actionable points.',s) for s in excerpts]
    summary=ai.llm('Create summary, chapter headings and show notes from these summaries. Do not invent timestamps.', '\n'.join(notes))
    return {'transcript':transcript,'segments':segments,'show_notes':summary}


run = podcast
