import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

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
    # TODO-BLOCK-BEGIN
    m,n = kernel.shape[0:2]
    h,w = img.shape[0:2]
    rdif = int((m-1)/2)
    cdif = int((n-1)/2)
    ht = h + 2*rdif
    wt = w + 2*cdif
    if img.ndim == 2:
        imgtmp = np.zeros((ht,wt))
    elif img.ndim == 3 and img.shape[2] == 3:
        imgtmp = np.zeros((ht,wt,3))
        kernel = kernel.reshape((m,n,1)) + np.zeros((1,3))
    else:
        raise Exception("TODO in hybrid.py not implemented")
    imgtmp[rdif:h+rdif,cdif:w+cdif] = img
    imgout = np.array([[(imgtmp[i:i+2*rdif+1,j:j+2*cdif+1]*kernel).sum(axis=0).sum(axis=0) for j in range(w)] for i in range(h)])
    return imgout
    # TODO-BLOCK-END

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
    # TODO-BLOCK-BEGIN
    kernel_corre = kernel[::-1,::-1]
    imgout = cross_correlation_2d(img,kernel_corre)
    if imgout.shape != img.shape :
        raise Exception("TODO in hybrid.py not implemented")
    else:
        return imgout
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    gaussian_func = lambda sigma,x,y:(1/(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2*(sigma**2)))
    kernel = np.array([[gaussian_func(sigma,i,j) for j in range(-width//2+1,width//2+1)] for i in range(-height//2+1,height//2+1)])
    return kernel/kernel.sum()
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return convolve_2d(img,gaussian_blur_kernel_2d(sigma,size,size))
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return img-low_pass(img,sigma,size)
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

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
   #return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
    min_h=np.amin(hybrid_img)
    max_h=np.amax(hybrid_img)
    return ((hybrid_img-min_h)/(max_h-min_h) * 255).clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    img_left = cv2.imread('left.png')
    #imgout1 = convolve_2d(img_left,gaussian_blur_kernel_2d(6.0,10,10)
    #For this problem, kernels with even size are prohibited but feasible in gui.py.
    #print(imgout1)
    img_right = cv2.imread('right.png')
    hyimg = create_hybrid_image(img_left,img_right,3.0,9,'low',6.0,13,'high',0.5)
    cv2.imwrite('mixed.png',hyimg,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
    print('images mixed.')
