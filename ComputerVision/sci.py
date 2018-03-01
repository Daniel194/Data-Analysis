from PIL import Image
from numpy import *
from scipy.ndimage import filters
from matplotlib import pyplot as plt

### Blurring ###

im = array(Image.open('image/images.jpeg'))

im2 = zeros(im.shape)

for i in range(3):
    im2[:, :, i] = filters.gaussian_filter(im[:, :, i], 3)

im2 = uint8(im2)

plt.imshow(im2)
plt.show()

### Derivatives ###

im = array(Image.open('image/images.jpeg').convert('L'))

# Sobel derivative filter

imx = zeros(im.shape)
filters.sobel(im, 1, imx)

imy = zeros(im.shape)
filters.sobel(im, 0, imy)

magnitude = sqrt(imx ** 2 + imy ** 2)

magnitude = uint8(magnitude)

plt.imshow(magnitude, cmap='gray')
plt.show()
