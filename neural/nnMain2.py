from nn01 import neuralNetwork
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance

#入力層、隠れ層、出力層のノード数
input_nodes=784
hidden_nodes=200
output_nodes=10

#学習率
learning_rate=0.1

#ニューラルネットワークのインスタンス生成
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#MNIST訓練データのCSVファイルを読み込んでリストにする
training_data_file=open("mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()

#ニューラルネットワークの学習

#訓練データが学習で使われた回数
epochs=2

#訓練データの全データに対して実行
for e in range(epochs):
	for record in training_data_list:
		all_values=record.split(',')
		inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
		#教師配列の生成(ラベルの位置が0.99残りは0.01)
		targets=numpy.zeros(output_nodes)+0.01
		targets[int(all_values[0])]=0.99
		n.train(inputs,targets)
		pass
	pass


#ニューラルネットワークのテスト

img=Image.open("image_3.png")
img= img.convert('L').resize((28,28),Image.LANCZOS)
contrast_converter = ImageEnhance.Contrast(img)
contrast_img = contrast_converter.enhance(6.0)

img_array = numpy.array(contrast_img)
# 28*28から784の長さのリストにする
img_data  = 255.0 - numpy.reshape(img_array,784)

# 入力値のスケーリングとシフト
img_data = (img_data / 255.0 * 0.99) + 0.01
print("min = ", numpy.min(img_data))
print("max = ", numpy.max(img_data))

# 画像出力
plt.imshow(contrast_img, cmap='gray')


#ネットワークを照会
outputs = n.query(img_data)
print (outputs)
label=numpy.argmax(outputs)
print(label,"network's answer")
plt.show()
