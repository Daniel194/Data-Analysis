from PIL import Image
from pylab import *
from numpy import *
import imtools

im = array(Image.open('image/images.jpeg').convert('L'))

figure()
gray()

contour(im, origin='image')

axis('equal')
axis('off')

figure()

hist(im.flatten(), 128)

show()

################################################################

im2, cdf = imtools.histeq(im)

figure()
gray()

contour(im2, origin='image')

axis('equal')
axis('off')

figure()

hist(im2.flatten(), 128)

show()