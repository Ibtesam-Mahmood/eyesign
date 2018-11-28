import numpy as np
import cv2

from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.activations import  linear
from keras import *
from keras.models import model_from_json

from keras.preprocessing import image
import matplotlib.pyplot as pyplot
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping

cap = cv2.VideoCapture(0)

model = Sequential()

model.add(InputLayer(input_shape=[64,64,1]))
model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(29, activation='softmax'))
optimizer = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

def get_guess_text(label):
    if label == 1:
        one_hot = "A"
    elif label == 2:
        one_hot = "B"
    elif label == 3:
        one_hot = "C"
    elif label == 4:
        one_hot = "D"
    elif label == 5:
        one_hot = "E"
    elif label == 6:
        one_hot = "F"
    elif label == 7:
        one_hot = "G"
    elif label == 8:
        one_hot = "H"
    elif label == 9:
        one_hot = "I"
    elif label == 10:
        one_hot = "J"
    elif label == 11:
        one_hot = "K"
    elif label == 12:
        one_hot = "L"
    elif label == 13:
        one_hot = "M"
    elif label == 14:
        one_hot = "N"
    elif label == 15:
        one_hot = "O"
    elif label == 16:
        one_hot = "P"
    elif label == 17:
        one_hot = "Q"
    elif label == 18:
        one_hot = "R"
    elif label == 19:
        one_hot = "S"
    elif label == 20:
        one_hot = "T"
    elif label == 21:
        one_hot = "U"
    elif label == 22:
        one_hot = "V"
    elif label == 23:
        one_hot = "W"
    elif label == 24:
        one_hot = "X"
    elif label == 25:
        one_hot = "Y"
    elif label == 26:
        one_hot = "Z"
    elif label == 27:
        one_hot = "space"
    elif label == 29:
        one_hot = "del"
    else:
        one_hot = "space"
    return one_hot


def getMaxIndex(array):
    max = 0
    index = -1
    maxIndex = 0
    for i in array[0]:
        index = index + 1
        if (i >= max):
            max = i
            maxIndex = index
    return maxIndex

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(frame, (64, 64))
    nparray = np.array(gray)

    newArray = nparray.reshape(-1,64,64,1)

    guess = model.predict(newArray)
    print(guess)
    print(getMaxIndex(guess))
    # print(get_guess_text(getMaxIndex(guess) ))

    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


