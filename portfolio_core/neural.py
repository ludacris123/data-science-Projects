"""TensorFlow inference; no randomly initialized classifier is served."""
import base64
import io
import json
import os
from pathlib import Path
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def load_artifact(task):
    import tensorflow as tf
    directory=Path(os.environ.get('MODEL_DIR','models'))/task
    if not (directory/'model.keras').exists() or not (directory/'labels.json').exists():
        raise ValueError(f'Train {task} first: model.keras and labels.json are required in {directory}')
    return tf.keras.models.load_model(directory/'model.keras'),json.loads((directory/'labels.json').read_text())


def image_array(encoded):
    import tensorflow as tf
    raw=base64.b64decode(encoded,validate=True)
    if len(raw)>10_000_000: raise ValueError('Image limit: 10 MB')
    return tf.image.resize(tf.io.decode_image(raw,channels=3,expand_animations=False),[224,224]).numpy()[None,...]


def medical(data):
    import tensorflow as tf
    model,labels=load_artifact('medical-image-classifier')
    x=image_array(data['image_base64'])
    probabilities=model(x,training=False).numpy()[0]
    # Training model exposes a top-level convolution block for Grad-CAM.
    layer=model.get_layer('cam_conv')
    grad_model=tf.keras.Model(model.inputs,[layer.output,model.output])
    with tf.GradientTape() as tape:
        convolution,preds=grad_model([x])
        index=tf.argmax(preds[0]); loss=preds[:,index]
    gradient=tape.gradient(loss,convolution)
    weights=tf.reduce_mean(gradient,axis=(0,1,2))
    heatmap=tf.reduce_sum(convolution[0]*weights,axis=-1)
    heatmap=tf.maximum(heatmap,0)/(tf.reduce_max(heatmap)+1e-8)
    return {'probabilities':dict(zip(labels,map(float,probabilities))),'heatmap':heatmap.numpy().tolist(),'label':labels[int(index)],'notice':'Research model. Scores are not calibrated diagnostic probabilities.'}


def sign(data):
    model,labels=load_artifact('sign-language-translator')
    landmarks=np.asarray(data.get('landmarks',[]),dtype='float32')
    if landmarks.shape!=(30,126): raise ValueError('Expected 30 frames × 126 features (two hands × 21 landmarks × xyz)')
    probabilities=model(landmarks[None,...],training=False).numpy()[0]
    return {'word':labels[int(np.argmax(probabilities))], 'probabilities':dict(zip(labels,map(float,probabilities)))}


def audio_features(raw):
    import librosa
    audio,sr=librosa.load(io.BytesIO(raw),sr=22050,duration=30)
    if len(audio)<22050: raise ValueError('At least one second of audio required')
    spec=librosa.power_to_db(librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=128),ref=np.max)
    spec=(spec+80)/80
    spec=spec[:,:1292]
    spec=np.pad(spec,((0,0),(0,max(0,1292-spec.shape[1]))))
    return spec.astype('float32')[None,...,None]


def music(data):
    raw=base64.b64decode(data['audio_base64'],validate=True)
    if len(raw)>20_000_000: raise ValueError('Audio limit: 20 MB')
    model,labels=load_artifact('music-genre-mood-classifier')
    features=audio_features(raw)
    probabilities=model(features,training=False).numpy()[0]
    return {'label':labels[int(np.argmax(probabilities))],'probabilities':dict(zip(labels,map(float,probabilities))),'spectrogram':features[0,::4,::16,0].tolist()}


@lru_cache(maxsize=2)
def image_encoder(kind):
    if kind=='clip':
        import keras_hub
        return (keras_hub.models.CLIPBackbone.from_preset('clip_vit_base_patch32'),keras_hub.models.CLIPPreprocessor.from_preset('clip_vit_base_patch32'))
    import tensorflow as tf
    return tf.keras.applications.EfficientNetB0(weights='imagenet',include_top=False,pooling='avg')


def visual(data):
    import faiss
    catalog=data.get('catalog',[])
    if not 1<=len(catalog)<=30: raise ValueError('Provide 1–30 catalog images')
    images=np.concatenate([image_array(i['image_base64']) for i in catalog])
    use_clip=bool(data.get('query')) or data.get('encoder')=='clip'
    if use_clip:
        import tensorflow as tf
        model,preprocessor=image_encoder('clip')
        processed=preprocessor({'prompts':['catalog']*len(images),'images':images})
        vectors=model.get_vision_embeddings(processed['images']).numpy()
        if data.get('query'):
            processed_query=preprocessor({'prompts':[data['query']],'images':images[:1]})
            query=model.get_text_embeddings(processed_query['token_ids']).numpy()
        else:
            processed_query=preprocessor({'prompts':['query'],'images':image_array(data['image_base64'])})
            query=model.get_vision_embeddings(processed_query['images']).numpy()
    else:
        encoder=image_encoder('efficientnet')
        vectors=encoder(images,training=False).numpy()
        query=encoder(image_array(data['image_base64']),training=False).numpy()
    vectors=np.ascontiguousarray(vectors,dtype='float32');query=np.ascontiguousarray(query,dtype='float32')
    faiss.normalize_L2(vectors);faiss.normalize_L2(query)
    index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors)
    scores,ids=index.search(query,min(10,len(catalog)))
    return {'matches':[{'id':catalog[int(i)]['id'],'similarity':float(score)} for i,score in zip(ids[0],scores[0])],'encoder':'TensorFlow KerasHub CLIP' if use_clip else 'TensorFlow EfficientNetB0'}

SERVICES={'medical-image-classifier':medical,'sign-language-translator':sign,'music-genre-mood-classifier':music,'visual-search-engine':visual}
