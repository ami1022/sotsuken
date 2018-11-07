import numpy as np
from PIL import Image

file_name=input("Please Enter File Name:")
file_name2=input("Please Enter WaterMarking:")

im = np.array(Image.open(file_name).convert('L').resize((256, 256)))

num = input("Please Enter plane number: ")
x=int(num)

mark = np.array(Image.open(file_name2).convert('L').resize((256, 256)))
mark=(mark>128)

pic=0
im_bin = {}

for i in range(0,8):
	if i==x:
		im_bin[i]=mark
	else:
		im_bin[i]=(im%2==1)
	im=im//2
	pic=pic+im_bin[i]*np.power(2,i)

Image.fromarray(np.uint8(pic)).save('out.png')