from numpy import *
from PIL import Image
from pylab import *


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weights=100):
    """
    An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    using the numerical procedure presentend in eq(11) A. Chambolle (2005).
    :param im: noisey inpute image (grayscale)
    :param U_init: initial guess for U
    :param tolerance: tolerance for stope criterion
    :param tau: steplength
    :param tv_weights: the TV-regularizing term
    :return: denoised and detextured image, texture residual
    """

    m, n = im.shape

    # initialize
    U = U_init
    Px = im
    Py = im
    error = 1

    while (error > tolerance):
        Uold = U

        GradUx = roll(U, -1, axis=1) - U
        GradUy = roll(U, -1, axis=0) - U

        PxNew = Px + (tau / tv_weights) * GradUx
        PyNew = Py + (tau / tv_weights) * GradUy
        NormNew = maximum(1, sqrt(PxNew ** 2 + PyNew ** 2))

        RxPx = roll(Px, 1, axis=1)
        RyPy = roll(Py, 1, axis=0)

        DivP = (Px - RxPx) + (Py - RyPy)
        U = im + tv_weights + DivP

        error = linalg.norm(U - Uold) / sqrt(n * m)

    return U, im - U


im = array(Image.open('image/images.jpeg').convert('L'))

U, T = denoise(im, im)

figure()
gray()

imshow(U)
axis('equal')
axis('off')
show()
