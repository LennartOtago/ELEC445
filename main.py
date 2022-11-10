import numpy as np
from PIL import Image, ImageOps
from numpy import array
import cv2
import scipy
from scipy import signal
from scipy.fftpack import fft2, ifft2, fftshift, fftn, ifftn, ifftshift
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

D1 = [[None], [None], [None]]

L = 10
K = 10
M = 10

print('singular values bigger than 10^-6')

for (i, M) in zip([0, 1, 2], [10, 30, 100]):
    D1[i] = np.zeros((M, M))
    for l in range(0, M):
        for k in range(0, M):
            j = 10 #can be 1,2,10
            D1[i][l][k] = M / (j * (l + 1) + j * (k + 1))

    w, v = np.linalg.eig(D1[i])
    s = np.sqrt(w)

    print(np.sum(s>1e-6))

# ################################################################


gray_img = mpimg.imread('jupiter1.tif')


# get psf from satellite
org_img = array(gray_img)
#plt.imshow(org_img, cmap='gray')
#plt.show()

xpos = 234
ypos = 85  # Pixel at centre of satellite
sat_img = org_img[ypos - 16: ypos + 16, xpos - 16:xpos + 16]
sat_img = sat_img / (sum(sum(sat_img)))

#plt.imshow(sat_img, cmap='gray')
#plt.show()

# # blurred image
# A_blurred = org_img * 1 / 25
# img = Image.fromarray(A_blurred)
#
# zero pad image so padded_img has same size as org_img
#padded_img = np.pad(sat_img, ((69, 155), (218, 6)))
padded_img = np.pad(sat_img, ((112,112), (112,112)) )

# plt.imshow(padded_img, cmap='gray')
# plt.show()


#
# # # convolved image with modified with normalized psf
# H = fft2(padded_img)
# F = fft2(org_img)
# A_conv = ifft2(np.multiply(F, H))
#

# # G = np.multiply(F,H)
# # # g = np.fft.fftshift(np.fft.ifftn(G).real)
A_conv = signal.convolve2d(org_img, sat_img, mode='same')
# #A_conv = ifft2(fft2(scipy.signal.convolve2d(org_img, sat_img, mode = 'same')))
# # # A_conv = cv2.filter2D(org_img, ddepth = -1, kernel=sat_img)
#
# # # A_conv = scipy.ndimage.convolve(org_img, sat_img )#, mode='constant', cval=0.0)
# # # img = Image.fromarray(g)

#plt.imshow(A_conv, cmap='gray')
#plt.show()


# naive inversion by direct division
fourier_img = fftshift(fft2(padded_img))
four_conv = fftshift(fft2(A_conv))
decov_img = abs(fftshift(ifft2(ifftshift(np.divide(four_conv, fourier_img)))))

#plt.imshow(decov_img, cmap='gray')
#plt.show()


#
# # fourier transfrom padded image
# fourier_img = fft2(padded_img)
# #img = Image.fromarray(fftshift(fourier_img).real)
# # Y = np.abs(np.fft.ifft2(1/fourier_img))
# # img.show()
#
# # get singular values
# # magnitude of finite fourier transform in descending order
# magn_psf = np.abs(fourier_img)
# sorted_psf = np.sort(magn_psf, axis=None)
# fig, ax = plt.subplots()
# ax.plot(range(0, len(sorted_psf)), sorted_psf[::-1])
# ax.set_yscale('log')
# # plt.show()
# plt.close()
#
# # plot magnitude
# x = np.linspace(0, 255, 256)
# y = np.linspace(0, 255, 256)
# X, Y = np.meshgrid(x, y)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(X, Y, magn_psf)
# # fig.show()
# plt.close()
#


norm_psf = abs(fourier_img)**2

lam = 0
tikh_img = ifftshift( ifft2( fftshift( four_conv * np.conj(fourier_img) / (lam ** 2 + norm_psf))))
tikh_img = abs(tikh_img)
#plt.imshow(tikh_img, cmap='gray')
#plt.show()


lambas = [1e-4, 1e-3, 1e-3, 0.1, 0.5, 0.8, 1, 10, 100, 1000]

norm_f = [None] * len(lambas)
norm_data = [None] * len(lambas)

for i in range(0, len(lambas)):
    tikh_img = ifftshift(ifft2(fftshift(four_conv * np.conj(fourier_img) / (lambas[i] ** 2 + norm_psf))))
    tikh_img = abs(tikh_img)
    norm_f[i] = np.sqrt(sum(sum(np.multiply(tikh_img, tikh_img))))
    D = tikh_img - A_conv
    norm_data[i] = np.sqrt(sum(sum(np.multiply(D,D))))

fig2 = plt.figure()
ax = fig2.add_subplot()
ax.set_yscale('log')
ax.set_xscale('log')

plt.plot(norm_data, norm_f)
for i, txt in enumerate(lambas):
    ax.annotate(txt, (norm_data[i], norm_f[i]))
plt.show()


# G = np.fft.fftn(A_conv)
#
# F_hat = np.divide(G, H)
# f_hat = np.fft.fftshift(np.fft.ifftn(F_hat).real)
# # f_hat = np.fft.ifftn(F_hat).real
# img = Image.fromarray(f_hat)
# # img.show()
#
# # X= np. multiply(A_conv)
# #
# (lam**2 + F**2)


# # this works
# # convolve image to blurr
# H = np.fft.fftn(padded_img)
# F = np.fft.fftn(org_img)
# D = np.multiply(F, H)
# d = np.fft.fftshift(np.fft.ifftn(D).real)
# # img = Image.fromarray(g)
# # img.show()
#
# # recover image by naive deconvolution
# D = np.fft.fftn(d)
# F_hat = np.divide(D, H)
# f_hat = np.fft.fftshift(np.fft.ifftn(F_hat).real)
# img = Image.fromarray(f_hat)
# img.show()
#
# lam = 1.3
# norm_psf = lam ** 2 + np.multiply(H, np.conj(H))
# X = np.multiply(D, np.conj(H))
# tikh_img = np.divide(X, norm_psf)
# img = Image.fromarray(np.fft.fftshift(np.fft.ifftn(tikh_img).real))
# img.show()
