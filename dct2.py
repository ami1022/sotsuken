#!/bin/env python
import numpy as np
import scipy.misc
from scipy.fftpack import dct, idct
import sys
from PIL import Image

H = 128
W = 128

img=Image.open("images/LENNA.png")
lenna = scipy.misc.imresize(img, (H, W)).astype(float)
lenna_F = dct(dct(lenna, axis=0), axis=1) ## 2D DCT of lenna

canvas = np.zeros((H,W))
for h in range(H):
    for w in range(W):
        a = np.zeros((H,W))
        a[h,w] = 1
        base = idct(idct(a, axis=0), axis=1) ## create dct bases
        canvas += lenna_F[h,w] * base ## accumulate
        if w==1:
        	#scipy.misc.imsave("base-%03d-%03d.png" % (h, w), base)
       		scipy.misc.imsave("lenna-%03d-%03d.png" % (h, w), canvas)