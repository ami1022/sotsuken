'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
from __future__ import print_function
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

import numpy as np


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

encoding_dim = 8  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(9,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(10, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
import copy

center=350
train_num=1000
test_num=10

batch_size = 64
epochs = 50


arr = np.random.rand(train_num)
arr2 = np.random.rand(test_num)
embedded_bit_train=np.round(arr)
embedded_bit_test=np.round(arr2)


(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

decoded_img = copy.deepcopy(x_test)
encoded_img = copy.deepcopy(x_test)

train_list=[[0 for i in range(10)] for j in range(train_num)]
test_list=[[0 for i in range(10)] for j in range(test_num)]

#mnistに透かし埋め込む箇所を決めてその周囲の画素値を入力データにする。

for y in range(train_num):
    del train_list[y][0:10]
    train_list[y].extend(x_train[y][center-28-1:center-28+2])
    train_list[y].extend(x_train[y][center-1:center+2])
    train_list[y].extend(x_train[y][center+28-1:center+28+2])
    train_list[y].append(embedded_bit_train[y])

for y in range(test_num):
    del test_list[y][0:10]
    test_list[y].extend(x_test[y][center-28-1:center-28+2])
    test_list[y].extend(x_test[y][center-1:center+2])
    test_list[y].extend(x_test[y][center+28-1:center+28+2])
    test_list[y].append(embedded_bit_test[y])

train_list=np.array(train_list)
test_list=np.array(test_list)

y_train = copy.deepcopy(train_list)
y_test = copy.deepcopy(test_list)

y_train = np.delete(y_train, 9, 1)
y_test = np.delete(y_test, 9, 1)

model1 = Sequential()
model1.add(Dense(9, activation='relu', input_shape=(10,)))
model1.add(Dropout(0.2))
model1.add(Dense(9, activation='sigmoid'))
model1.add(Dropout(0.2))

model2 = Sequential()
model2.add(Dense(9, activation='relu', input_shape=(10,)))
model2.add(Dropout(0.2))
model2.add(Dense(9, activation='sigmoid'))
model2.add(Dropout(0.2))
#model.add(Dense(num_classes, activation='softmax'))

#model.summary()

#encoder
model1.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model2.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model1.fit(train_list, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(train_list,y_train))

history = model2.fit(train_list, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(test_list,y_test))
#score = model1.evaluate(test_list, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

train_data=model1.predict(train_list)
test_data=model2.predict(test_list)

#autoencoder
autoencoder.fit(train_data, train_list,
                epochs=200,
                batch_size=64,
                shuffle=True,
                validation_data=(test_data, test_list))




import matplotlib.pyplot as plt


# テスト画像を変換
#print(test_list[0])
encoded_data = encoder.predict(test_list)
#print(encoded_data[0])
decoded_data = decoder.predict(encoded_data)
#print(decoded_data[0])


for i in range(test_num):
    encoded_img[i][center-28-1:center-28+2]=encoded_data[i][0:3]
    encoded_img[i][center-1:center+2]=encoded_data[i][3:6]
    encoded_img[i][center+28-1:center+28+2]=encoded_data[i][6:9]

for i in range(test_num):
    decoded_img[i][center-28-1:center-28+2]=decoded_data[i][0:3]
    decoded_img[i][center-1:center+2]=decoded_data[i][3:6]
    decoded_img[i][center+28-1:center+28+2]=decoded_data[i][6:9]
    print(i,",",decoded_data[i][9],embedded_bit_test[i])


# 何個表示するか
n = test_num
plt.figure(figsize=(20, 10))
for i in range(n):
    # オリジナルのテスト画像を表示
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(encoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 変換された画像を表示
    ax = plt.subplot(3, n, i+1+(2*n))
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()