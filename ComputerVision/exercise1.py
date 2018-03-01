from PIL import Image
from numpy import *
from scipy.ndimage import filters
from matplotlib import pyplot as plt

### 2 ###

im_gray = array(Image.open('image/images.jpeg').convert('L'))
img_color = array(Image.open('image/images.jpeg'))

for i in range(3):
    img_color[:, :, i] -= filters.gaussian_filter(img_color[:, :, i], 100)

img_color = uint8(img_color)

plt.imshow(img_color)
plt.show()

im_gray -= filters.gaussian_filter(im_gray, 100)

plt.imshow(im_gray, cmap='gray')
plt.show()

### 3 ###

im_gray = array(Image.open('image/images.jpeg').convert('L'))

im_quatient = im_gray / filters.gaussian_filter(im_gray, 100)

plt.imshow(im_quatient, cmap='gray')
plt.show()
