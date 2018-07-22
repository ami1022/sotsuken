from nn01 import neuralNetwork
import numpy

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
scorecard=[]

test_data_file=open("mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()

for record in test_data_list:
	all_values=record.split(',')
	#正解は配列の1番目
	correct_label=int(all_values[0])
	#print(correct_label,"correct label")
	#入力値のスケーリングとシフト
	inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
	#ネットワークへの照会
	outputs=n.query(inputs)
	#最大値のインデックスがラベルに対応
	label=numpy.argmax(outputs)
	#print(label,"network's answer")

	#正解(1),間違い(0)をリストに追加
	if(label==correct_label):
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass
	pass

#print(scorecard)

scorecard_array=numpy.asarray(scorecard)
print("performance=",scorecard_array.sum()/scorecard_array.size)