import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')
#sys.path.append('/usr/local/opt/opencv3/lib/python2.7/site-packages')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    m, n = kernel.shape
    k1, k2 = (m - 1) / 2, (n - 1) / 2
    a = b = c = 1
    if len(img.shape) > 2:
        a, b, c = img.shape
        img_padd = np.zeros((a + 2 * k1, b + 2 * k2, c))
        img_padd[k1: k1 + a, k2: k2 + b, :] = img
        ret = np.zeros((a, b, c))
        for i in xrange(a):
            for j in xrange(b):
                tmp = np.zeros((c,))
                for u in xrange(-k1, k1 + 1):
                    for v in xrange(-k2, k2 + 1):
                        tmp += img_padd[i + k1 + u, j + k2 + v, :] * kernel[u + k1, v + k2]
                ret[i, j] = tmp
        return ret
    else:
        a, b = img.shape
        img_padd = np.zeros((a + 2 * k1, b + 2 * k2))
        img_padd[k1: k1 + a, k2: k2 + b] = img
        ret = np.zeros((a, b))
        for i in xrange(a):
            for j in xrange(b):
                tmp = 0
                for u in xrange(-k1, k1 + 1):
                    for v in xrange(-k2, k2 + 1):
                        tmp += img_padd[i + k1 + u, j + k2 + v] * kernel[u + k1, v + k2]
                ret[i, j] = tmp
        return ret

def kernel_flip(kernel):
    '''A helper method of convolve_2d
    '''
    return np.fliplr(np.flipud(kernel))

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img, kernel_flip(kernel))

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    offset_x, offset_y = (width - 1) / 2, (height - 1) / 2
    x = np.arange(-offset_x, offset_x + 1, 1.0) ** 2
    y = np.arange(-offset_y, offset_y + 1, 1.0) ** 2
    coefficient = 1 / (2 * sigma * sigma * np.pi)
    gaussian_x = np.sqrt(coefficient) * np.exp(-x / (2 * sigma * sigma))
    gaussian_y = np.sqrt(coefficient) * np.exp(-y / (2 * sigma * sigma))
    # in two dimensions, it is the product of gaussian_x and gaussian_y, one in each dimension
    kernel = np.outer(gaussian_x, gaussian_y) / (np.sum(gaussian_x) * np.sum(gaussian_y))
    return kernel

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)