"""medical-image-classifier: project-specific business logic."""
import base64
import numpy as np
from portfolio_core import neural

def medical(data):
    import tensorflow as tf
    model,labels=neural.load_artifact('medical-image-classifier')
    x=neural.image_array(data['image_base64'])
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


run = medical


def build_model(classes):
    import tensorflow as tf
    inputs = tf.keras.Input((224,224,3))
    encoder = tf.keras.applications.EfficientNetB0(include_top=False,weights='imagenet',input_shape=(224,224,3))
    encoder.trainable = False
    features = encoder(inputs,training=False)
    features = tf.keras.layers.Conv2D(128,1,activation='relu',name='cam_conv')(features)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    return tf.keras.Model(inputs,tf.keras.layers.Dense(classes,activation='softmax')(features))
