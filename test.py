import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance

img=Image.open("LENNA.png")
img= img.resize((28,28),Image.LANCZOS)
contrast_converter = ImageEnhance.Contrast(img)
contrast_img = contrast_converter.enhance(6.0)
plt.imshow(contrast_img, cmap='gray')
plt.show()