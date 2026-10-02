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
    from .registry import load_service
    return load_service('medical-image-classifier').run(data)


def sign(data):
    from .registry import load_service
    return load_service('sign-language-translator').run(data)


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
    from .registry import load_service
    return load_service('music-genre-mood-classifier').run(data)


@lru_cache(maxsize=2)
def image_encoder(kind):
    if kind=='clip':
        import keras_hub
        return (keras_hub.models.CLIPBackbone.from_preset('clip_vit_base_patch32'),keras_hub.models.CLIPPreprocessor.from_preset('clip_vit_base_patch32'))
    import tensorflow as tf
    return tf.keras.applications.EfficientNetB0(weights='imagenet',include_top=False,pooling='avg')


def visual(data):
    from .registry import load_service
    return load_service('visual-search-engine').run(data)

SERVICES={'medical-image-classifier':medical,'sign-language-translator':sign,'music-genre-mood-classifier':music,'visual-search-engine':visual}
