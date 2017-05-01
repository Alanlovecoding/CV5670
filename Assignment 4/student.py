# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    height, width, channel = images[0].shape[0], images[0].shape[1], images[0].shape[2]
    albedo = np.zeros((height, width, channel), dtype = np.float32)
    normals = np.zeros((height, width, 3), dtype = np.float32)
    n_images = len(images)

    for i in range(height):
        for j in range(width):
            for c in range(channel):
                I = np.array([image[i,j,c] for image in images])
                I.reshape((n_images, 1))
                L = lights.T
                G = np.dot(np.linalg.inv(np.dot(L.T, L)), np.dot(L.T, I))
                kd = np.linalg.norm(G)
                if kd < 1e-7:
                    kd, G_ = 0, np.zeros((3,1))
                else:
                    G_ = G / np.linalg.norm(G)
                albedo[i, j, c] = kd
                normals[i, j] = G_.reshape((3,))

    return albedo, normals


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    height, width = points.shape[0], points.shape[1]
    ret = np.zeros((height, width, 2))

    # precompute K*Rt
    krt = np.dot(K, Rt)

    for i in range(height):
        for j in range(width):
            p = points[i,j]
            p = np.append(p, 1)
            _p = np.dot(krt, p)
            ret[i,j,0] = _p[0] / _p[2]
            ret[i,j,1] = _p[1] / _p[2]
    return ret


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x112, x121, x122, x211, x212, x221, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height, width, channels = image.shape
    normalized = np.zeros((height, width, channels * ncc_size * ncc_size), dtype = np.float32)

    ncc_half= ncc_size / 2

    for i in range(height):
        for j in range(width):
            if i - ncc_half < 0 or i + ncc_half >= height or j - ncc_half < 0 or j + ncc_half >= width:
                continue
            tmp = list()
            for c in range(channels):
                win = image[i - ncc_half: i + ncc_half + 1, j - ncc_half: j + ncc_half + 1, c]
                win = (win - np.mean(win)).flatten()
                tmp.append(win.T)
            vec = np.hstack(tuple(tmp)).flatten()
            nn = np.linalg.norm(vec)
            if nn < 1e-6:
                vec.fill(0)
            else:
                vec = vec / nn
            normalized[i,j] = vec
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height, width = image1.shape[0], image1.shape[1]
    ncc = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            ncc[i,j] = np.correlate(image1[i,j], image2[i,j])[0]
    return ncc
