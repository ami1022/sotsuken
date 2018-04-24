import numpy as np
from PIL import Image

file_name=input("Please Enter File Name:")
file_name2=input("Please Enter WaterMarking:")

im = np.array(Image.open(file_name).convert('L').resize((256, 256)))
#print(type(im))

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


#pic=im_bin[0]+im_bin[1]*2+im_bin[2]*4+im_bin[3]*8+im_bin[4]*16+im_bin[5]*32+im_bin[6]*64+im_bin[7]*128+mark

Image.fromarray(np.uint8(pic)).save('out.png')