"""Train TensorFlow classifiers on explicit train/val/test splits; export held-out metrics."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    import tensorflow as tf
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['medical-image-classifier','sign-language-translator','music-genre-mood-classifier'])
    p.add_argument('--data',required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--output',default='models')
    args=p.parse_args(); tf.keras.utils.set_random_seed(42)
    root=Path(args.data)
    if args.task=='medical-image-classifier':
        train=tf.keras.utils.image_dataset_from_directory(root/'train',image_size=(224,224),batch_size=16)
        labels=train.class_names
        val=tf.keras.utils.image_dataset_from_directory(root/'val',class_names=labels,image_size=(224,224),batch_size=16,shuffle=False)
        test=tf.keras.utils.image_dataset_from_directory(root/'test',class_names=labels,image_size=(224,224),batch_size=16,shuffle=False)
        from .registry import load_service
        model=load_service(args.task).build_model(len(labels))
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(train,validation_data=val,epochs=args.epochs,callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])
        metrics=model.evaluate(test,verbose=0,return_dict=True)
    else:
        # Separate patient/speaker/song sources BEFORE preparing the NPZ splits.
        splits=[np.load(root/(s+'.npz')) for s in ['train','val','test']]
        labels=json.loads((root/'labels.json').read_text())
        x,y=splits[0]['x'],splits[0]['y']
        from .registry import load_service
        model=load_service(args.task).build_model(len(labels))
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(x,y,validation_data=(splits[1]['x'],splits[1]['y']),epochs=args.epochs,batch_size=16,callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])
        metrics=model.evaluate(splits[2]['x'],splits[2]['y'],verbose=0,return_dict=True)
    target=Path(args.output)/args.task; target.mkdir(parents=True,exist_ok=True)
    model.save(target/'model.keras')
    (target/'labels.json').write_text(json.dumps(labels))
    (target/'metrics.json').write_text(json.dumps({k:float(v) for k,v in metrics.items()}))

if __name__=='__main__': main()
