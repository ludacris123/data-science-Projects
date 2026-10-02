"""music-genre-mood-classifier: project-specific business logic."""
import base64
import numpy as np
from portfolio_core import neural

def music(data):
    raw=base64.b64decode(data['audio_base64'],validate=True)
    if len(raw)>20_000_000: raise ValueError('Audio limit: 20 MB')
    model,labels=neural.load_artifact('music-genre-mood-classifier')
    features=neural.audio_features(raw)
    probabilities=model(features,training=False).numpy()[0]
    return {'label':labels[int(np.argmax(probabilities))],'probabilities':dict(zip(labels,map(float,probabilities))),'spectrogram':features[0,::4,::16,0].tolist()}


run = music


def build_model(classes):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.Input((128,1292,1)),
        tf.keras.layers.Conv2D(16,3,activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32,3,activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(classes,activation='softmax'),
    ])
