"""visual-search-engine: project-specific business logic."""
import base64
import numpy as np
from portfolio_core import neural

def visual(data):
    import faiss
    catalog=data.get('catalog',[])
    if not 1<=len(catalog)<=30: raise ValueError('Provide 1–30 catalog images')
    images=np.concatenate([neural.image_array(i['image_base64']) for i in catalog])
    use_clip=bool(data.get('query')) or data.get('encoder')=='clip'
    if use_clip:
        import tensorflow as tf
        model,preprocessor=neural.image_encoder('clip')
        processed=preprocessor({'prompts':['catalog']*len(images),'images':images})
        vectors=model.get_vision_embeddings(processed['images']).numpy()
        if data.get('query'):
            processed_query=preprocessor({'prompts':[data['query']],'images':images[:1]})
            query=model.get_text_embeddings(processed_query['token_ids']).numpy()
        else:
            processed_query=preprocessor({'prompts':['query'],'images':neural.image_array(data['image_base64'])})
            query=model.get_vision_embeddings(processed_query['images']).numpy()
    else:
        encoder=neural.image_encoder('efficientnet')
        vectors=encoder(images,training=False).numpy()
        query=encoder(neural.image_array(data['image_base64']),training=False).numpy()
    vectors=np.ascontiguousarray(vectors,dtype='float32');query=np.ascontiguousarray(query,dtype='float32')
    faiss.normalize_L2(vectors);faiss.normalize_L2(query)
    index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors)
    scores,ids=index.search(query,min(10,len(catalog)))
    return {'matches':[{'id':catalog[int(i)]['id'],'similarity':float(score)} for i,score in zip(ids[0],scores[0])],'encoder':'TensorFlow KerasHub CLIP' if use_clip else 'TensorFlow EfficientNetB0'}


run = visual
