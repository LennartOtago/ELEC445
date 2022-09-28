import numpy as np
from PIL import Image, ImageOps
from numpy import array
import cv2

import matplotlib.pyplot as plt

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
for (i, M) in zip([0,1,2], [10, 30, 100]):
    D1[i] = np.zeros((M,M))
    for l in  range(0,M):
        for k in  range(0,M):
            j = 2
            D1[i][l][k] = M/(j* (l+1) + j* (k +1) )

    u, s, vh = np.linalg.svd(D1[i])
    #print(np.sum(s>1e-6))


im = Image.open('jupiter1.tif')
#im.show()

#print(im.size)

gray_img = ImageOps.grayscale(im)
#gray_img.show()
A = array(gray_img)

#Z = np.zeros((256,256))

#blurred image
A_blurred = A *1/25
img = Image.fromarray(A_blurred)
img.show()
xpos = 234
ypos = 85 # Pixel at centre of satellite

print((xpos-16, 256-(xpos + 16) ))
print(ypos-16, 256-(ypos + 16) )




sat_img = A[ypos-16: ypos + 16, xpos-16 :xpos+16]

#sat_img = sat_img/np.amax(sat_img)
sat_img = sat_img/sum(sum(sat_img))

#padded_A = np.pad(A[ypos-16: ypos + 16, xpos-16 :xpos+16], ((69,155), (218,6)) )

padded_img = np.pad(sat_img, ((69,155), (218,6)) )

#img = Image.fromarray(padded_img)
#img.show()



fourier_img = np.fft.fft2(padded_img)
print(fourier_img[0,20])

magn_psf = np.abs(fourier_img)


print(np.amax(np.abs(fourier_img)))
x= np.linspace(0,255,256)
y= np.linspace(0,255,256)

X,Y = np.meshgrid(x,y)

#fig = plt.figure()
#ax = fig.add_subplot(projection = '3d')
#ax.plot_surface(X,Y, magn_psf)
#fig.show()



decov_img = np.fft.ifft2(np.fft.fft2(A)/fourier_img)
img = Image.fromarray(np.abs(decov_img))
#img.show()


#img = Image.fromarray(sat_img)
#img.show()


#magnitude of finite fourier transform in descending order

sorted_psf = np.sort(magn_psf, axis=None)
fig, ax = plt.subplots()
ax.plot(range(0,len(sorted_psf)), sorted_psf[::-1])
ax.set_yscale('log')
#plt.show()

print(sorted_psf)

test = np.abs(fourier_img * np.conj(fourier_img))

lam = 0.8
tikh_img = np.fft.ifft2(np.fft.fft2(A) * np.conj(fourier_img)/(lam**2 + test ) )
img = Image.fromarray(np.abs(tikh_img))
img.show()
#h = h./sum(sum(h));