from scipy.ndimage import filters
from numpy import *
from pylab import *
from PIL import Image


def compute_harris_response(im, sigma=3):
    """
    Compute the Harris corer detector response function
    for each pixel in a grayscale image
    :param im:
    :param sigma:
    :return:
    """

    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)

    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, treshold=0.1):
    """
    :param harrisim: the image
    :param min_dist: is the minimum number of pixels separating corners and image boundary
    :param treshold:
    :return: return corners from a Harris response image
    """

    corner_threshold = harrisim.max() * treshold
    harrisim_t = (harrisim > corner_threshold) * 1

    coords = array(harrisim_t.nonzero()).T

    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    index = argsort(candidate_values)

    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    filtered_coords = []

    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    figure()
    gray()

    imshow(image)

    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')

    axis('off')
    show()


def get_descriptors(image, filtered_coords, wid=5):
    """
    For each point return pixel values around the point using a neighbourhood of width 2*wid+1
    ( Assume points are extracted with min_distance > wid).
    :param image:
    :param filtered_coords:
    :param wid:
    :return:
    """

    desc = []

    for coords in filtered_coords:
        path = image[coords[0] - wid:coords[0] + wid + 1, coords[1] - wid:coords[1] + wid + 1].flatten()
        desc.append(path)

    return desc


def match(desc1, desc2, treshold=0.5):
    """
    For each corner point descriptor in the first image, select its match to second image using
    normalized croos correlation
    :param desc1:
    :param desc2:
    :param treshold:
    :return:
    """

    n = len(desc1[0])

    d = - ones((len(desc1), len(desc2)))

    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n - 1)

            if ncc_value > treshold:
                d[i, j] = ncc_value

    ndx = argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores


def match_twosided(desc1, desc2, trashold=0.5):
    """
    Two sided symetric version of match()
    :param desc1:
    :param desc2:
    :param trashold:
    :return:
    """

    matches_12 = match(desc1, desc2, trashold)
    matches_21 = match(desc2, desc1, trashold)

    ndx_12 = where(matches_12 >= 0)[0]

    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12


im = array(Image.open('image/images.jpeg').convert('L'))
harrisim = compute_harris_response(im)
filtered_coords = get_harris_points(harrisim, 6)
plot_harris_points(im, filtered_coords)


def appendimages(im1, im2):
    # select the imagewith the fewest rows abd fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2, zeros((rows1 - rows2, im2.shape[1]))), axis=0)


def plot_mathces(im1, im2, locs1, locs2, matchscores, show_belwo=True):
    im3 = append(im2, im2)

    if show_belwo:
        im3 = vstack((im3, im3))

    imshow(im3)

    cols1 = im1.shape[1]

    for i, m in enumerate(matchscores):
        if m > 0:
            plot([locs1[i][1], locs1[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
            axis('off')
