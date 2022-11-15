import random

import numpy as np
import pylab as pl
from PIL import Image, ImageOps
from numpy import array
import cv2
import scipy
from scipy import signal
from scipy.fftpack import fft2, ifft2, fftshift, fftn, ifftn, ifftshift
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

import numpy.random as rd

import function
#

# m1 = 10
# m2 = 30
# m3 = 100
#
# x1 = [i+1 for i in range(m1)]
# x2 = [i+1 for i in range(m2)]
# x3 = [i+1 for i in range(m3)]
#
# y1 = [ np.exp( -1*x/m1) for x in x1]
# y2 = [ np.exp( -1*x/m2) for x in x2]
# y3 = [ np.exp( -1*x/m3) for x in x3]
#
# plt.figure(1)
# plt.plot(x1,y1)
# plt.plot(x2,y2)
# plt.plot(x3,y3)
#
#
#
# y1 = [ np.exp( -2*x/m1) for x in x1]
# y2 = [ np.exp( -2*x/m2) for x in x2]
# y3 = [ np.exp( -2*x/m3) for x in x3]
#
# plt.figure(2)
# plt.plot(x1,y1)
# plt.plot(x2,y2)
# plt.plot(x3,y3)
#
#
# y1 = [ np.exp( -10*x/m1) for x in x1]
# y2 = [ np.exp( -10*x/m2) for x in x2]
# y3 = [ np.exp( -10*x/m3) for x in x3]
#
# plt.figure(3)
# plt.plot(x1,y1)
# plt.plot(x2,y2)
# plt.plot(x3,y3)
#
# plt.show()

##############################################################3

# D1 = [[None], [None], [None]]
#
# L = 10
# K = 10
# M = 10
#
# print('singular values bigger than 10^-6')
#
# for (i, M) in zip([0, 1, 2], [10, 30, 100]):
#     D1[i] = np.zeros((M, M))
#     for l in range(0, M):
#         for k in range(0, M):
#             j = 10 #can be 1,2,10
#             D1[i][l][k] = M / (j * (l + 1) + j * (k + 1))
#
#     w, v = np.linalg.eig(D1[i])
#     s = np.sqrt(w)
#
#     print(np.sum(s>1e-6))

# ################################################################


# gray_img = mpimg.imread('jupiter1.tif')
#
#
# # get psf from satellite
# org_img = array(gray_img)
# plt.imshow(org_img, cmap='gray')
# #plt.show()
#
# xpos = 234
# ypos = 85  # Pixel at centre of satellite
# sat_img = org_img[ypos - 16: ypos + 16, xpos - 16:xpos + 16]
# sat_img = sat_img / (sum(sum(sat_img)))
#
# #plt.imshow(sat_img, cmap='gray')
# #plt.show()
#
# # # blurred image
# # A_blurred = org_img * 1 / 25
# # img = Image.fromarray(A_blurred)
# #
# # zero pad image so padded_img has same size as org_img
# #padded_img = np.pad(sat_img, ((69, 155), (218, 6)))
# padded_img = np.pad(sat_img, ((112,112), (112,112)) )
#
# # plt.imshow(padded_img, cmap='gray')
# # plt.show()
#
#
#
# # # # convolved image with modified with normalized psf
# A_conv = signal.convolve2d(org_img, sat_img, mode='same')
# plt.imshow(A_conv, cmap='gray')
# #plt.show()
#
#
# # naive inversion by direct division
# fourier_img = fftshift(fft2(padded_img))
# four_conv = fftshift(fft2(A_conv))
# decov_img = abs(fftshift(ifft2(ifftshift(np.divide(four_conv, fourier_img)))))
#
# #plt.imshow(decov_img, cmap='gray')
# #plt.show()
#
#
#
#
# norm_psf = abs(fourier_img)**2
#
# lam = 0.10019
# tikh_img = ifftshift( ifft2( fftshift( four_conv * np.conj(fourier_img).transpose() / (lam ** 2 + norm_psf))))
# tikh_img = abs(tikh_img)
# plt.imshow(tikh_img, cmap='gray')
# plt.show()
#
#
# lambas = [1e-4, 1e-3, 1e-3, 0.1, 0.5, 0.8, 1, 10, 100, 1000]
# lambas = np.linspace(0,10,100)
#
# norm_f = [None] * len(lambas)
# norm_data = [None] * len(lambas)
#
# for i in range(0, len(lambas)):
#     tikh_img = ifftshift(ifft2(fftshift(four_conv * np.conj(fourier_img).transpose() / (lambas[i] ** 2 + norm_psf))))
#     tikh_img = abs(tikh_img)
#     norm_f[i] = np.sqrt(sum(sum(np.multiply(tikh_img, tikh_img))))
#     FH = ifftshift(ifft2(fftshift(fourier_img * fftshift(fft2(tikh_img)) )))
#     D = abs(FH - A_conv)
#     norm_data[i] = np.sqrt(sum(sum(np.multiply(D,D))))
#
# #plt.imshow(FH.real, cmap='gray')
# #plt.show()
#
#
# fig2 = plt.figure()
# ax = fig2.add_subplot()
# ax.set_xscale('log')
# ax.set_yscale('log')
# print(lambas[::100])
#
# plt.scatter(norm_data, norm_f)
# for i, txt in enumerate(lambas[0:20]):
#     ax.annotate(np.around(txt,5), (norm_data[i], norm_f[i]))
# plt.show()

#####################################################################

#mcmc for exponential distribution
#n = int(1e3)  # length of markov chain
#w = float(1)  # width of random walk


#X = function.mcexp(n,w)

# fig1, ax1 = plt.subplots(3)
# X = function.mcgaus(0, 1, 1, int(1e3), 0.3)
# ax1[0].plot(X)
# X = function.mcgaus(0,1, 1, int(1e3), 3)
# ax1[1].plot(X)
# X = function.mcgaus(0, 1, 1, int(1e3), 30)
# ax1[2].plot(X)
# plt.show()



# X = function.mcgaus(0, 1, 20, int(1e3), 3)
# plt.plot(X)
# plt.show()
#
# #pause
# #burn in
#
# X = function.mcgaus(0, 1, 1, int(1e6), 3)
#
# plt.hist(X, bins=30)
# plt.show()
#
# print(np.var(X))
# print(np.mean(X))


######## compute 2 ##########

# ws = np.linspace(1,8,100)
# mu = 0
# sig = 1
# x0 = 0
# n = int(1e5)
#
# for cnt in range(0,len(ws)):
#     w = ws[cnt]
#     X = function.mcgaus(mu, sig, x0, n, w)
#     value, dvalue, ddvalue, tauint, dtauint, Qval = function.UWerr(X.transpose(), 1.5, length(X), 0)
#     taus[cnt] = tauint * 2
#


# D = 0.5
# L = 10
# T = 2
#
# alpha = np.exp(-D* T* (np.pi/L)**2)
# l = np.linspace(1,15,15)
#
# y = alpha**(l**2)
#
# #fig = plt.figure()
# plt.scatter(l,y)
# plt.yscale('log')
# plt.xlabel('index')
# plt.ylabel('eigenvalue')
# plt.show()



############ compute 4 ############

npix = 100
slide = np.ones((npix, npix)) #make npix*npix image

ncellmax = 10   #max no of cells
cellrad = 9.5   #radius of cells
stddev = 0.1 # noise standard deviation

ncell = 5 + math.ceil((ncellmax - 5) * rd.rand()) #random number of cells at leas 5
pbad = 0.25 + 0.5* rd.rand()        #probability a cell is bad
A = npix * rd.rand(2, ncell)
xycell = np.ceil(npix * rd.rand(2, ncell))

for icell in range(0,ncell-1):
    if rd.rand() < pbad:
        slide = function.putbad(slide, xycell[0,icell], xycell[1,icell], cellrad)
    else:
        slide = function.putgood(slide, xycell[0,icell], xycell[1,icell], cellrad)

x,y = slide.shape
slide = slide + stddev * rd.rand(x,y)

plt.imshow(slide,cmap='gray')
plt.show()















print("bla")