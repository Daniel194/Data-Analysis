import os
import numpy as np
from PIL import Image


def get_imlist(path):
    """
    :param path: the absolute path to a folder.
    :return: return a list of filenames for jpg images in a directory.
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    """
    :param im: the image
    :param sz: the size of the new image
    :return: the image im with the size sz
    """
    pil_im = Image.fromarray(np.uint8(im))

    return np.array(pil_im.resize(sz))


def histeq(im, nr_bins=256):
    """
    Histogram equalization of a grayscale image
    :param im: the image
    :param nr_bins: the number of segmentation
    :return: the histogram equalization of a grayscale image
    """

    # get the image histogram
    imhist, bins = np.histogram(im.flatten(), nr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


def compute_average(imlist):
    """
    Compute the average of a list of images
    :param imlist: A list of images
    :return: return the average of a list of images
    """

    averageim = np.array(Image.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print(imname + '..... skipped')

    averageim /= len(imname)

    return np.array(averageim, 'uint8')
