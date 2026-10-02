"""Validate neural plumbing with disposable test models, never clinical weights."""
import base64
import json
import numpy as np
import pytest
from portfolio_core import neural

tf=pytest.importorskip('tensorflow')


def test_medical_artifact_and_gradcam(tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_DIR',str(tmp_path));neural.load_artifact.cache_clear()
    directory=tmp_path/'medical-image-classifier';directory.mkdir()
    inputs=tf.keras.Input((224,224,3));x=tf.keras.layers.Rescaling(1/255)(inputs)
    x=tf.keras.layers.Conv2D(2,3,activation='relu',name='cam_conv')(x)
    model=tf.keras.Model(inputs,tf.keras.layers.Dense(3,activation='softmax')(tf.keras.layers.GlobalAveragePooling2D()(x)))
    model.save(directory/'model.keras');(directory/'labels.json').write_text(json.dumps(['normal','pneumonia','other']))
    encoded=base64.b64encode(tf.io.encode_png(tf.ones((224,224,3),dtype=tf.uint8)*100).numpy()).decode()
    result=neural.medical({'image_base64':encoded})
    assert abs(sum(result['probabilities'].values())-1)<1e-5
    heatmap=np.asarray(result['heatmap']);assert heatmap.shape==(222,222) and np.isfinite(heatmap).all()
    neural.load_artifact.cache_clear()


def test_sign_real_keras_artifact(tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_DIR',str(tmp_path));neural.load_artifact.cache_clear()
    directory=tmp_path/'sign-language-translator';directory.mkdir()
    model=tf.keras.Sequential([tf.keras.Input((30,126)),tf.keras.layers.LSTM(4),tf.keras.layers.Dense(2,activation='softmax')]);model.save(directory/'model.keras')
    (directory/'labels.json').write_text('["A","B"]')
    result=neural.sign({'landmarks':np.zeros((30,126)).tolist()})
    assert result['word'] in ['A','B'] and abs(sum(result['probabilities'].values())-1)<1e-5
    neural.load_artifact.cache_clear()


def test_music_spectrogram():
    import io,soundfile as sf
    out=io.BytesIO();sf.write(out,np.sin(np.arange(22050)*.1),22050,format='WAV')
    features=neural.audio_features(out.getvalue())
    assert features.shape==(1,128,1292,1) and np.isfinite(features).all()


def test_missing_trained_artifact(tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_DIR',str(tmp_path));neural.load_artifact.cache_clear()
    with pytest.raises(ValueError,match='Train'):neural.load_artifact('medical-image-classifier')
