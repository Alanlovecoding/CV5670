### hybrid.py

In hybrid.py, we are asked to implement 4 function.

* cross_correlation_2d(img, kernel)
* convolve_2d(img, kernel)
* gaussian_blur_kernel_2d(sigma, width, height)
* low_pass(img, sigma, size)
* high_pass(img, sigma, size)

#### cross_correlation_2d

Given a kernel of arbitrary m x n dimensions, with both m and n being odd, compute the cross correlation of the given image with the given kernel, such that the output is of the same dimension as the image and assume the pixels out of the bounds of the image to be zero.

Input image can be either RGB image or a grayscale image as a numpy array.

to compute 2d cross_correlation, the first thing to do is to padding zeros. In this function i used img2col method to convert the correlation operation to matrix multiplication.

#### convolve_2d

2d convolution can be carried out by using the `cross_correlation_2d()`. just flip the input kernel upside down, and then flip it in left/right direction. Then put the modified kernel into the cross_correlation_2d.

#### gaussian_blur_kernel_2d

this function return a gaussian blur kernel of the given dimensions and with the given sigma.

* sigma: controls the radius of the Gaussian blur. In this case, width == height.
* width: width of the kernel
* height: height of the kernel

two dimension gaussian is the product of two 1_d gaussian.

so in this function, I compute the outer product of  the gaussian_x and gaussian_y as two 1_d gaussian

#### low_pass

low_pass function filter the image with a low_pass filter. as a result, it suppressed the higher frequency components. In order to filter an image with low_pass filter, just need to compute the convolution of the image with the given 2-d gaussian_blur_kernel.

#### high_pass

high_pass function filter the image with a high pass filter. in other words, it keep the high frequency part of the image.

in order to get this part, we just need to subtract the low_pass part of the image from the original image.

