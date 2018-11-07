
from __future__ import print_function
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import copy

import matplotlib.pyplot as plt

center=350
train_num=1000
test_num=10

batch_size = 64
epochs = 100

#ステゴ画像生成ニューラルネットワークモデル生成
#model1:autoencoderの学習用ステゴ画像
#model2:autoencoderのテスト用ステゴ画像

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

model1.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model2.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#autoencoderモデル生成
encoding_dim = 9  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_img = Input(shape=(10,))
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
decoded = Dense(10, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


#入力データ形成
#透かしビット系列作成
arr = np.random.rand(train_num)
arr2 = np.random.rand(test_num)
embedded_bit_train=np.round(arr)
embedded_bit_test=np.round(arr2)

#データダウンロード
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#後で画像表示する用
decoded_img = copy.deepcopy(x_test)
encoded_img = copy.deepcopy(x_test)

#LENNA読み込み
import cv2
img = cv2.imread('LENNA.png')
img = img.astype('float32') / 255.
img = x_train.reshape((len(img), np.prod(img.shape[1:])))

#mnistに透かし埋め込む箇所を決めてその周囲の画素値を入力データにする。
train_list=[[0 for i in range(10)] for j in range(train_num)]
test_list=[[0 for i in range(10)] for j in range(test_num)]
lenna_list=[]

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

lenna_list[y].extend(img[center-28-1:center-28+2])
lenna_list[y].extend(img[center-1:center+2])
lenna_list[y].extend(x_train[y][center+28-1:center+28+2])
lenna_list[y].append(1)

train_list=np.array(train_list)
test_list=np.array(test_list)

#ステゴ画像作るときのラベルがわり
y_train = copy.deepcopy(train_list)
y_test = copy.deepcopy(test_list)
y_train = np.delete(y_train, 9, 1)
y_test = np.delete(y_test, 9, 1)

#いざ学習
#原画像入れてステゴ画像出す
#10→9
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

train_data=model1.predict(train_list)
train_data2 = copy.deepcopy(train_data)
test_data=model2.predict(test_list)
test_data2 = copy.deepcopy(test_data)
print("長さ",len(train_data[2]))

#autoencoderに入力できるように長さ揃える
train_data2=np.insert(train_data, 9, 0, axis=1)
test_data2=np.insert(test_data, 9, 0, axis=1)

#autoencoder
autoencoder.fit(train_list, train_list,
                nb_epoch=200,
                batch_size=64,
                shuffle=True,
                validation_data=(test_list, test_list))

output = encoder.predict(train_list)
output=np.insert(output, 9, 0, axis=1)
train_data2=(output+train_data2)/2

output2=encoder.predict(test_list)
output2=np.insert(output2, 9, 0, axis=1)
test_data2=(output2+test_data2)/2

autoencoder.fit(train_data2, train_list,
                epochs=200,
                batch_size=64,
                shuffle=True,
                validation_data=(test_data2, test_list))

encoded_data = encoder.predict(test_data2)
decoded_data = decoder.predict(encoded_data)


#画像の画素値置き換え
for i in range(test_num):
    encoded_img[i][center-28-1:center-28+2]=test_data[i][0:3]
    encoded_img[i][center-1:center+2]=test_data[i][3:6]
    encoded_img[i][center+28-1:center+28+2]=test_data[i][6:9]

for i in range(test_num):
    decoded_img[i][center-28-1:center-28+2]=decoded_data[i][0:3]
    decoded_img[i][center-1:center+2]=decoded_data[i][3:6]
    decoded_img[i][center+28-1:center+28+2]=decoded_data[i][6:9]
    print(i,",",decoded_data[i][9],embedded_bit_test[i])

#画像表示
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