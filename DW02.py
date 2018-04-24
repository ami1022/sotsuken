import numpy as np
from PIL import Image

file_name=input("Please Enter File Name:")
file_name2=input("Please Enter file Name2:")

im = np.array(Image.open(file_name).convert('L').resize((256, 256)))
print(type(im))

im_bin_1=(im%2==1)*255
im_bin_2=((im//2)%2==1)*255
im_bin_3=(((im//2)//2)%2==1)*255
im_bin_4=((((im//2)//2)//2)%2==1)*255
im_bin_5=(((((im//2)//2)//2)//2)%2==1)*255
im_bin_6=((((((im//2)//2)//2)//2)//2)%2==1)*255
im_bin_7=(((((((im//2)//2)//2)//2)//2)//2)%2==1)*255
im_bin_8=((((((((im//2)//2)//2)//2)//2)//2)//2)%2==1)*255

im_bin = np.concatenate((im_bin_1,im_bin_2,im_bin_3,im_bin_4,im_bin_5,im_bin_6,im_bin_7,im_bin_8), axis=1)
Image.fromarray(np.uint8(im_bin)).save(file_name2)