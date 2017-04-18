import math
import sys

import cv2
import numpy as np
from numpy.linalg import inv


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def iround(x):
    if x < 0.0:
        return int(x - 0.5)
    else:
        return int(x + 0.5)


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    width, height = img.shape[1], img.shape[0]
    ul = np.array([[0, 0, 1]]).T
    ur = np.array([[width, 0, 1]]).T
    dl = np.array([[0, height, 1]]).T
    dr = np.array([[width, height, 1]]).T
    ul_t = np.dot(M, ul).T
    ur_t = np.dot(M, ur).T
    dl_t = np.dot(M, dl).T
    dr_t = np.dot(M, dr).T
    ul_t[0, 0] /= ul_t[0, 2]
    ul_t[0, 1] /= ul_t[0, 2]
    ur_t[0, 0] /= ur_t[0, 2]
    ur_t[0, 1] /= ur_t[0, 2]
    dl_t[0, 0] /= dl_t[0, 2]
    dl_t[0, 1] /= dl_t[0, 2]
    dr_t[0, 0] /= dr_t[0, 2]
    dr_t[0, 1] /= dr_t[0, 2]
    minX = iround(min(ul_t[0, 0], ur_t[0, 0], dl_t[0, 0], dr_t[0, 0]))
    minY = iround(min(ul_t[0, 1], ur_t[0, 1], dl_t[0, 1], dr_t[0, 1]))
    maxX = iround(max(ul_t[0, 0], ur_t[0, 0], dl_t[0, 0], dr_t[0, 0]))
    maxY = iround(max(ul_t[0, 1], ur_t[0, 1], dl_t[0, 1], dr_t[0, 1]))
    #TODO-BLOCK-END
    return minX, minY, maxX, maxY


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    width = img.shape[1]
    height = img.shape[0]
    min_x, min_y, max_x, max_y = imageBoundingBox(img, M)
    # lumaScale = 1.0
    # cnt = 0
    #
    # for ii in range(min_x, max_x-1):
    #     for jj in range(min_y, max_y-1):
    #         flag = False
    #         p = np.array([[ii, jj, 1]]).T
    #         p = np.dot(inv(M), p)
    #         newx = int(p[0][0] / p[2][0])
    #         newy = int(p[1][0] / p[2][0])
    #         if newx >= 0 and newx < width and newy >= 0 and newy < height:
    #             if acc[jj, ii, 0] == 0 and acc[jj, ii, 1] == 0 and acc[jj, ii, 2] == 0:
    #                 flag = True
    #             if img[newy, newx, 0] == 0 and img[newy, newx, 1] == 0 and img[newy, newx, 2] == 0:
    #                 flag = True
    #             if not flag:
    #                 lumaAcc = 0.299 * acc[jj, ii, 0] + 0.587 * acc[jj, ii, 1] + 0.114 * acc[jj, ii, 2]
    #                 lumaImg = 0.299 * img[newy, newx, 0] + 0.587 * img[newy, newx, 1] + 0.114 * img[newy, newx, 2]
    #
    #                 if lumaImg != 0:
    #                     scale = lumaAcc / lumaImg
    #                     if scale > 0.5 and scale < 2:
    #                         lumaScale += lumaAcc / lumaImg
    #                         cnt += 1
    #
    # if cnt != 0:
    #     lumaScale = lumaScale / float(cnt)
    # else:
    #     lumaScale = 1.0
    #
    # weight = 0.0

    for ii in range(min_x, max_x):
        for jj in range(min_y, max_y):
            p = np.array([[ii, jj, 1]]).T
            p = np.dot(inv(M), p)
            newx = int(p[0][0] / p[2][0])
            newy = int(p[1][0] / p[2][0])
            if newx >= 0 and newx < width - 1 and newy >= 0 and newy < height - 1:
                weight = 1.0
                c1, c2 = 2**31, 2**31
                if ii >= min_x and ii < min_x + blendWidth:
                    c1 = float(ii - min_x) / blendWidth
                if ii <= max_x and ii > max_x - blendWidth:
                    c2 = float(max_x - ii) / blendWidth
                weight = min(c1, weight, c2)
                if img[newy, newx, 0] == 0 and img[newy, newx, 1] == 0 and img[newy, newx, 2] == 0:
                    weight = 0.0

                R = img[newy, newx, 0]
                G = img[newy, newx, 1]
                B = img[newy, newx, 2]

                # r = 255.0 if R * lumaScale > 255.0 else R * lumaScale
                # g = 255.0 if G * lumaScale > 255.0 else G * lumaScale
                # b = 255.0 if B * lumaScale > 255.0 else B * lumaScale

                acc[jj, ii, 0] += R * weight
                acc[jj, ii, 1] += G * weight
                acc[jj, ii, 2] += B * weight
                acc[jj, ii, 3] += weight
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    height = acc.shape[0]
    width = acc.shape[1]
    img = np.zeros((height, width, 3))
    for ii in range(height):
        for jj in range(width):
            if acc[ii,jj,3] > 0:
                img[ii, jj, 0] = int(acc[ii, jj, 0] / acc[ii, jj, 3])
                img[ii, jj, 1] = int(acc[ii, jj, 1] / acc[ii, jj, 3])
                img[ii, jj, 2] = int(acc[ii, jj, 2] / acc[ii, jj, 3])
    img = np.uint8(img)
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        _minX, _minY, _maxX, _maxY = imageBoundingBox(img, M)
        minX = min(minX, _minX)
        minY = min(minY, _minY)
        maxX = max(maxX, _maxX)
        maxY = max(maxY, _maxY)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    k = 0
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

