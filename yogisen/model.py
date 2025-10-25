# Train enhanced model with 3D landmarks + features
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import numpy as np
import os
from database import DATA_DIR

path = os.path.join(DATA_DIR,'assembled_dataset_3d.npz')
d = np.load(path)
X,y = d['X'],d['y']
idx = np.arange(len(X)); np.random.shuffle(idx)
X,y = X[idx],y[idx]
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

num_classes = len(np.unique(y))
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(1024,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256,activation='relu'),
    layers.Dense(num_classes,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=40,batch_size=32)
model.save(os.path.join(DATA_DIR,"yogisense_model_3d.h5"))
