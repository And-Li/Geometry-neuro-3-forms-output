from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import hw_light

base_dir = 'hw_light'
x_train = []
y_train = []
img_height = 20
img_width = 20

for patch in os.listdir(base_dir):
    for img in os.listdir(base_dir + '/' + patch):
        x_train.append(image.img_to_array(image.load_img(base_dir + '/' + patch + '/' + img,
                                                         target_size=(img_height, img_width),
                                                         color_mode='grayscale')))
        if patch == '0':
            y_train.append(0)
        elif patch == '3':
            y_train.append(1)
        else:
            y_train.append(2)

x_train = np.array(x_train)
y_train = np.array(y_train)

print('Размер массива x_train: ', x_train.shape)
print('Размер массива y_train: ', y_train.shape)

CLASS = 3  # setting the number of output classes (circle, triangle, square)
BATCH = (10, 100, 1000)  # set of batch_size changes in .fit
# reshaping the input datasets
x_train_ready = x_train.reshape(x_train.shape[0], -1)
y_train_ready = utils.to_categorical(y_train, CLASS)


# defining a function with two params we are to change
def create_model(activation_name, batch_size_number):
    net = Sequential()
    net.add(Dense(10, input_dim=400,
                  activation=activation_name))  # first dimension of tne input array gotta be equal to the input_dim!!!
    net.add(Dense(100, activation=activation_name))
    net.add(Dense(5000, activation=activation_name))
    net.add(Dense(CLASS, activation='softmax'))
    net.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.001),
                metrics=['accuracy'])

    net.fit(x_train_ready,
            y_train_ready,
            batch_size=batch_size_number,
            epochs=10,
            verbose=0)
    scores = net.evaluate(x_train_ready, y_train_ready, verbose=0)
    return scores[1]


# setting our neuromodels ready with the needed params:
net_1 = create_model('relu', 10)
net_2 = create_model('linear', 10)
net_3 = create_model('linear', 100)
net_4 = create_model('linear', 1000)

# getting out the results & filling the table to analyze the outputs:
df = pd.DataFrame(
    {
        "Net Name": ['net_1', 'net_2', 'net_3', 'net_4'],
        "Accuracy": [net_1, net_2, net_3, net_4],
        "Parameter Description": ['Dense 10-100-5000, relu, batch_size = 10',
                                  'Dense 10-100-5000, linear, batch_size = 10',
                                  'Dense 10-100-5000, linear, batch_size = 100',
                                  'Dense 10-100-5000, linear, batch_size = 1000'],
    }
)
print(df)
