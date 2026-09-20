#1)Download Dataset

#https://drive.google.com/drive/folders/19zV45_NQzrBPLzFymXeNZiufxbeENGth

#2) Unzip
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.utils import load_img
from keras.models import Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,GlobalMaxPooling2D

Base_dir='datasets/UTKFace/'
MODEL_PATH = 'models/model.keras'
if not os.path.exists(MODEL_PATH):
    images=os.listdir(Base_dir)
    image_paths = []
    age_labels = []
    gender_labels = []

    for filename in images:
        image_path = os.path.join(Base_dir, filename)
        temp = filename.split('_')
        if str(temp[0]).isnumeric():
            age = int(temp[0])
            gender = int(temp[1])
            image_paths.append(image_path)
            age_labels.append(age)
            gender_labels.append(gender)

    def make_dataset(images):
        features = []
        for image in images:
            img = load_img(image, color_mode='grayscale', target_size=(128, 128))
            img = np.array(img)
            features.append(img)

        features = np.array(features)
        features = features.reshape(len(features), 128, 128, 1)
        return features

    X = make_dataset(image_paths)
    X=X/255.0
    Y= np.array(age_labels)
if os.path.exists(MODEL_PATH):
    print(f'Loading existing model: {MODEL_PATH}')
    model = load_model(MODEL_PATH)
else:
    model= Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(128,128,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(GlobalMaxPooling2D())

    model.add(Dense(132 , activation='relu'))
    model.add(Dense(1))

    model.summary()
    model.compile(loss='mse', optimizer='adam' , metrics=['accuracy'] )
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
    X_train.shape
    model.fit(X_train,y_train,epochs=10, validation_data=(X_test,y_test))
    model.save_weights('models/model.weights.h5')
    model.save(MODEL_PATH)
detector = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)  # Try 1 if camera 0 does not work

if not cam.isOpened():
    raise RuntimeError("Could not open the camera")

while True:
    ret, frame = cam.read()

    if not ret:
        print("Could not read camera frame")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        face = face.astype("float32") / 255.0
        face = face.reshape(1, 128, 128, 1)

        age = model.predict(face, verbose=0)
        predicted_age = int(age[0][0])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"Age: {predicted_age}",
            (x, y - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Live", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()