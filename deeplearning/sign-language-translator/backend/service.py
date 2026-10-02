"""sign-language-translator: project-specific business logic."""
import base64
import numpy as np
from portfolio_core import neural

def sign(data):
    model,labels=neural.load_artifact('sign-language-translator')
    landmarks=np.asarray(data.get('landmarks',[]),dtype='float32')
    if landmarks.shape!=(30,126): raise ValueError('Expected 30 frames × 126 features (two hands × 21 landmarks × xyz)')
    probabilities=model(landmarks[None,...],training=False).numpy()[0]
    return {'word':labels[int(np.argmax(probabilities))], 'probabilities':dict(zip(labels,map(float,probabilities)))}


run = sign


def build_model(classes):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.Input((30,126)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(classes,activation='softmax'),
    ])
