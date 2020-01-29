import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATADIR = "dataset/training_set"
CATEGORIES = ["A","B","C","D","del","E","F","G","H","I","J","K","L","M","N",
              "nothing","O","P","Q","R","S","space","T","U","V","W","X","Y","Z"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break

print(img_array)
print(img_array.shape)

IMG_SIZE = 50
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#X = X / 255.0

import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out, protocol = 4)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out, protocol = 4)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X[1]
y[1]






import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPooling2D

NAME = "SLD-CNN-64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logsNew\{}'.format(NAME))

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

X = pickle.load(open("X50.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# Normalizing X
X = X/255.0

# Normalizing y
from tensorflow.keras.utils import to_categorical
y_binary = to_categorical(y)

# Model Training
model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape = X.shape[1:], activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(29, activation = 'softmax'))
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ["accuracy"])

model.fit(X, y_binary, batch_size = 64, epochs = 5, validation_split = 0.1, callbacks = [tensorboard])

# Saving Keras Model
#model.save("SLD-CNN-128-256-512-D-256-128-29.h5")

# Loading Saved Model
from tensorflow.keras.models import load_model
model = load_model("hash-file2.h5")
IMG_SIZE = 50
img_array = cv2.imread("B_test.jpg", cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#img = open(r"C:\Users\fluXcapacit0r\Documents\Python Scripts\SLD\dataset\test_set\A\A_test.jpg")
#img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#new_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

pred = model.predict([new_array])
print(pred[0][0])





import cv2
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPooling2D

#NAME = "SLD-CNN-64x2-{}".format(int(time.time()))
#tensorboard = TensorBoard(log_dir = 'logs\{}'.format(NAME))

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

from tensorflow.keras.utils import to_categorical
y_binary = to_categorical(y)

conv_layers = [1, 2, 3]
layer_sizes = [32, 64, 128]
dense_layers = [0, 1, 2]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir = 'logs\{}'.format(NAME))
            print(NAME)
            model = Sequential()
            model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:], activation = 'relu'))
            model.add(MaxPooling2D(pool_size = (2, 2)))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3), activation = 'relu'))
                model.add(MaxPooling2D(pool_size = (2, 2)))
            
            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size, activation = 'relu'))
            
            model.add(Dense(29, activation = 'softmax'))
            model.compile(optimizer = 'adam',
                          loss = 'categorical_crossentropy',
                          metrics = ["accuracy"])
            
            model.fit(X, y_binary, batch_size = 32, epochs = 10, validation_split = 0.1, callbacks = [tensorboard])


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Actual Processing of the image until camera is ON
while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        #results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    # When 'Q' is presses the pop-up will exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroying the Session
capture.release()
cv2.destroyAllWindows()
