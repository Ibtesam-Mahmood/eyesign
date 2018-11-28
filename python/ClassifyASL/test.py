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

# with open('model.json', 'r') as f:
#     json = f.read()
# model = model_from_json(json)

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

# model.add(InputLayer(batch_input_shape=(1, 5)))
# model.add(Dense(10, input_shape=(5,), activation=linear))
# model.compile(loss=losses.mean_squared_error,
#               optimizer=optimizers.sgd(),
#               metrics=[metrics.mean_absolute_error])
model.load_weights("eyesign.h5")

test_data = 'asl_alphabet_test'
A = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
B = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
C = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
D = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
E = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
F = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
G = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
H = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
I = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
J = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
K = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
L = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
M = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
N = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
O = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
P = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
K = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Q = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
R = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
S = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
T = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
U = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
W = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
X = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
Z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
delete = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
space = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
nothing = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def one_hot_label(img):
    label = img.split('.')[0]
    if label == "A":
        one_hot = A
    elif label == "B":
        one_hot = B
    elif label == "C":
        one_hot = C
    elif label == "D":
        one_hot = D
    elif label == "E":
        one_hot = E
    elif label == "F":
        one_hot = F
    elif label == "G":
        one_hot = G
    elif label == "H":
        one_hot = H
    elif label == "I":
        one_hot = I
    elif label == "J":
        one_hot = J
    elif label == "K":
        one_hot = K
    elif label == "L":
        one_hot = L
    elif label == "M":
        one_hot = M
    elif label == "N":
        one_hot = N
    elif label == "O":
        one_hot = O
    elif label == "P":
        one_hot = P
    elif label == "Q":
        one_hot = Q
    elif label == "R":
        one_hot = R
    elif label == "S":
        one_hot = S
    elif label == "T":
        one_hot = T
    elif label == "U":
        one_hot = U
    elif label == "V":
        one_hot = V
    elif label == "W":
        one_hot = W
    elif label == "X":
        one_hot = X
    elif label == "Y":
        one_hot = Y
    elif label == "Z":
        one_hot = Z
    elif label == "nothing":
        one_hot = nothing
    elif label == "space":
        one_hot = space
    elif label == "del":
        one_hot = delete
    return one_hot


def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        if path == "asl_alphabet_test/.DS_Store":
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        test_images.append([np.array(img), one_hot_label(i)])
    return test_images

def getMaxIndex(array):
    max = 0
    index = -1
    maxIndex = 0
    for i in array[0]:
        index = index + 1
        if(i >= max):
            max = i
            maxIndex = index
    return maxIndex

# print("% Accuracy: ", trues/29.0, "| Trues ", trues)
def print_percent(arr):
    for val in arr:
        print()

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

testing_images = test_data_with_label()

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,64,64,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

index = 0
trues = 0
def get_index(data, arr):
    index = -1
    for i in arr:
        index = index+1
        if i == data: return index
#
for cnt, data in enumerate(testing_images[0:28]):
    img = data[0]
    onehot = data[1]
    index = get_index(1, onehot)
    print("element ", onehot)
    data = img.reshape(1,64,64,1)
    guess = model.predict(data)

    print(guess)
    print(getMaxIndex(guess)," ?= ",index)
    if getMaxIndex(guess) == index:
        trues = trues+1
        print("True")
    else:
        print("False")
    print(get_guess_text(getMaxIndex(guess) + 1))
    print()

def setup():
    model = Sequential()

    model.add(InputLayer(input_shape=[64, 64, 1]))
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

    # model.add(InputLayer(batch_input_shape=(1, 5)))
    # model.add(Dense(10, input_shape=(5,), activation=linear))
    # model.compile(loss=losses.mean_squared_error,
    #               optimizer=optimizers.sgd(),
    #               metrics=[metrics.mean_absolute_error])
    model.load_weights("eyesign.h5")

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def getEstimate(arrayImage):
    img = arrayImage[0][0]
    img = img.copy()
    arrayImage = img.resize(1,64,64,1)
    guess = model.predict(arrayImage)
    print(get_guess_text(getMaxIndex(guess) + 1))



def getEst(nparray):
    img = nparray.copy()
    # onehot = data[1]
    # index = get_index(1, onehot)
    #     print("element ", onehot)
    data = img.reshape(1,64,64,1)
    guess = model.predict(data)
    print(guess)

