from __future__ import print_function
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import cifar10

import numpy as np
import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import copy

import matplotlib.pyplot as plt

import time

t1 = time.time()

center=16
train_num=500
test_num=1

batch_size = 128
encoded_epochs = 200
decoded_epochs = 100

#ステゴ画像生成ニューラルネットワークモデル生成

#autoencoderモデル生成
encoding_dim = 8  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_img = Input(shape=(9,))
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
decoded = Dense(10, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


x_test=cv2.imread("sea_stego.png")
x_test=cv2.cvtColor(x_test,cv2.COLOR_BGR2RGB).astype('float32')/255

#後で画像表示する用
decoded_img = copy.deepcopy(x_test)

#透かし埋め込む箇所を決めてその周囲の画素値を入力データにする。
y_test_r=copy.deepcopy(x_test[2000:2003,2000:2003,0:1]).reshape(test_num,3,3).reshape(test_num,9)
y_test_g=copy.deepcopy(x_test[2000:2003,2000:2003,1:2]).reshape(test_num,3,3).reshape(test_num,9)
y_test_b=copy.deepcopy(x_test[2000:2003,2000:2003,2:3]).reshape(test_num,3,3).reshape(test_num,9)


autoencoder.load_weights('decode.h5')

autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

encoded_data_r = encoder.predict(y_test_r)
decoded_data_r = decoder.predict(encoded_data_r)

encoded_data_g = encoder.predict(y_test_g)
decoded_data_g = decoder.predict(encoded_data_g)

encoded_data_b = encoder.predict(y_test_b)
decoded_data_b = decoder.predict(encoded_data_b)

print(decoded_data_r[0][9])
print(decoded_data_g[0][9])
print(decoded_data_b[0][9])

decoded_data_r=np.delete(decoded_data_r,9,1)
decoded_data_g=np.delete(decoded_data_g,9,1)
decoded_data_b=np.delete(decoded_data_b,9,1)

result2=np.sqrt(sum(np.square(decoded_data_r[0]*255-y_test_r[0]*255)))
print(1/(1 + result2))

decoded_img[2000:2003,2000:2003,0:1]=decoded_data_r.reshape(3,3,1)
decoded_img[2000:2003,2000:2003,1:2]=decoded_data_g.reshape(3,3,1)
decoded_img[2000:2003,2000:2003,2:3]=decoded_data_b.reshape(3,3,1)

#画像表示
n = test_num
i=0
plt.figure(figsize=(8, 5))
# オリジナルのテスト画像を表示
ax = plt.subplot(1, 3, i+1)
plt.imshow(x_test)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# 変換された画像を表示
ax = plt.subplot(1, 3, i+1+n)
plt.imshow(decoded_img)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)



decoded_img=cv2.cvtColor(decoded_img,cv2.COLOR_BGR2RGB)*255
cv2.imwrite("sea_result.png",decoded_img)

t2 = time.time()

# 経過時間を表示
elapsed_time = t2-t1
print(f"経過時間：{elapsed_time}")

plt.show()