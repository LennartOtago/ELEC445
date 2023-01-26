import numpy as np
import pylab as pl
from PIL import Image, ImageOps
from numpy import array
import scipy
from scipy import signal
from scipy.fftpack import fft2, ifft2, fftshift, fft, ifft, ifftshift
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
import sympy as sy
import numpy.random as rd
import numpy.linalg as lin
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


gray_img = mpimg.imread('jupiter1.tif')


# get psf from satellite
org_img = array(gray_img)
# plt.imshow(org_img, cmap='gray')
# plt.show()

xpos = 234
ypos = 85  # Pixel at centre of satellite
sat_img_org = org_img[ypos - 16: ypos + 16, xpos - 16:xpos + 16]
sat_img = sat_img_org / (sum(sum(sat_img_org)))

# plt.imshow(sat_img, cmap='gray')
# plt.show()

# # blurred image
# A_blurred = org_img * 1 / 25
# img = Image.fromarray(A_blurred)
#
# zero pad image so padded_img has same size as org_img
#padded_img = np.pad(sat_img, ((69, 155), (218, 6)))
padded_img = np.pad(sat_img, ((112,112), (112,112)) )

# plt.imshow(padded_img, cmap='gray')
# plt.show()



# # # convolved image with modified with normalized psf
# A_conv = signal.convolve2d(org_img, sat_img, mode='same')
# plt.imshow(A_conv, cmap='gray')
# plt.show()
#
#

# # naive inversion by direct division
fourier_img = fftshift(fft2(padded_img))
four_conv = fftshift(fft2(org_img))

decov_img = abs(ifftshift( ifft2( fftshift( np.divide(four_conv,fourier_img)) )))


print(sum(sum(org_img)))
print(sum(sum(decov_img)))
plt.imshow(abs(decov_img), cmap='gray')
plt.show()




# L_org = np.array([[ 0, -1, 0],[ -1, 4, -1],[ 0, -1, 0]])
# L =  np.pad(L_org, ((127,126), (127,126)) )
# #
# #
norm_psf = abs(fourier_img)**2
#
lam = 0.10019

tikh_img = ifftshift( ifft2( fftshift( four_conv * fourier_img / (lam ** 2 + norm_psf))))
tikh_img = abs(tikh_img)

plt.imshow(tikh_img, cmap='gray')
plt.show()
print(sum(sum(tikh_img)))
#


#
#
# #reg_img = ifftshift(ifft2(fftshift(four_conv * fourier_img / (norm_psf + alpha * abs(four_L) ) ) ))
#
# #plt.imshow(abs(reg_img), cmap='gray')
# #plt.show()
# #
# # XL = ifftshift( ifft2( fftshift( fftshift(fft2(reg_img)) * four_L)))
# # plt.imshow(abs(XL), cmap='gray')
# # plt.show()
#
# # lambas = [1e-4, 1e-3, 1e-3, 0.1, 0.5, 0.8, 1, 10, 100, 1000]
# alphas = np.linspace(0.000001,0.2,100)
# #alphas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) * 1e-3
# norm_f = [None] * len(alphas)
# norm_data = [None] * len(alphas)
#
# for i in range(0, len(alphas)):
#     four_L = fftshift(fft2(L))
# # np.conj(fftshift(fft2(padded_img.transpose() ))) * fftshift(fft2(org_img))
#     reg_img = ifftshift(ifft2(fftshift( four_conv * np.conj(fourier_img) /  (norm_psf +  alphas[i]  * abs(four_L)))))
#     #tikh_img = ifftshift(ifft2(fftshift(four_conv * fourier_img/ (lambas[i] ** 2 + norm_psf))))
#     #tikh_img = abs(tikh_img)
#     XL = ifftshift( ifft2( fftshift( fftshift(fft2(reg_img)) * four_L)))
#     norm_f[i] = np.sqrt(sum(sum(np.multiply(reg_img, abs(XL)))))
#     FH = ifftshift(ifft2(fftshift(fourier_img * fftshift(fft2(reg_img)) )))
#     D = abs(abs(FH) - org_img)**2
#     norm_data[i] = np.sqrt(sum(sum(D)))
#
# # #plt.imshow(FH.real, cmap='gray')
# #plt.show()
#
# #
# fig2 = plt.figure()
# ax = fig2.add_subplot()
# ax.set_xscale('log')
# ax.set_yscale('log')
# # print(lambas[::100])
#
#plt.scatter(norm_data, norm_f)
# for i, txt in enumerate(alphas[0:26]):
#    ax.annotate(np.around(txt,5), (norm_data[i], norm_f[i]))
#plt.show()




# res_img = ifftshift( ifft2( fftshift( gamma * four_conv * np.conj(fourier_img) + v_1 + v_2/ ( gamma *abs(fftshift(fft2(ATA))) +  rho * abs(four_L) ) )))

#plt.imshow(abs(res_img), cmap='gray')
#plt.show()



#sample from prior

l = np.array([[2,-1],[-1,2]])
L= np.zeros((258,258))

for i in range(0, len(L)-1):
        L[i:i+2,i:i+2] = L[i:i+2,i:i+2] + l

L =L[1:257,1:257]

L_org = np.array([[ 0, -1, 0],[ -1, 4, -1],[ 0, -1, 0]])

four_L = fft2( np.pad(L_org, ((127,126), (127,126)) ))

#norm_psf = fft2(np.multiply(padded_img.transpose(), padded_img) )

alpha = 0.0056724
gen_reg_img = abs(ifftshift( ifft2( fftshift( four_conv * np.conj(fourier_img) / (norm_psf + alpha * abs(four_L) )))) )
print(sum(sum(gen_reg_img)))

plt.imshow(gen_reg_img, cmap='gray')
plt.show()

a_gamma = 1
a_rho = 1
b_gamma = 1e-4
b_rho = 1e-4
m= 256**2

n_sample = 1000
res_img = np.zeros((n_sample,256,256))
rho =  np.zeros(n_sample)
gamma =  np.zeros(n_sample)


#intialize first sample

rho[0] = 5.16e-5

gamma[0] = 0.218

v_2= np.zeros(256**2).transpose()
[v_2[0] ,v_2[-1]] = [v_2[0] ,v_2[-1]] + rd.multivariate_normal([0,0],rho[0] *l)
for j in range(0, len(v_2)-1):
    [v_2[j],v_2[j+1]] =  [v_2[j],v_2[j+1]] + rd.multivariate_normal([0,0],rho[0] *l)


v_2 = v_2.reshape((256,256))
v_rd = np.multiply(rd.normal(0,1,256*256),1)
v_1= np.sqrt(gamma[0]) *  np.multiply(padded_img.transpose() ,v_rd.reshape((256,256)))
w = v_1 + v_2

res_img[0] = abs(ifftshift( ifft2( fftshift(  (  gamma[0]  *four_conv * np.conj(fourier_img) + fft2(w) )/ (gamma[0] * norm_psf + rho[0] * abs(four_L) ) ))))

plt.imshow(res_img[0], cmap='gray')
plt.show()
print(sum(sum(res_img[0])))
for i in range(1, n_sample-1):
    Ay = abs(ifftshift( ifft2( fftshift(  fft2(res_img[i - 1]) * fourier_img))))
    plt.imshow(abs(Ay), cmap='gray')
    plt.show()

    norm_res = sum(sum( ( abs(Ay) - org_img)**2))

    shape_gamma, scale_gamma = m/2 + a_gamma, 1/(0.5 * norm_res + b_gamma)

    gamma[i] = rd.gamma(shape_gamma, scale_gamma)
    shape_rho, scale_rho = m/2 + a_rho, 1/(0.5 * norm_res + b_rho)
    rho[i] = rd.gamma(shape_rho, scale_rho)

    v_1 = np.zeros(256).transpose()
    [v_1[0], v_1[-1]] = [v_1[0], v_1[-1]] + rd.multivariate_normal([0, 0], rho[i] *l)
    for j in range(0, len(L) - 1):
        [v_1[j], v_1[j + 1]] = [v_1[j], v_1[j + 1]] + rd.multivariate_normal([0, 0], rho[i] *l)
    v_rd = np.multiply(rd.normal(0, 1, 256), np.identity(256))
    v_2 = np.sqrt(gamma[i]) * np.multiply(padded_img.transpose(), v_rd)
    w = v_1 + v_2
    print(rho[i])
    print(gamma[i])
    res_img[i] = abs(ifft2(  (  gamma[i]  *four_conv * np.conj(fourier_img) + fft2(w) )/ (gamma[i] * norm_psf + rho[i] * four_L ) ))



G = sum(gamma[60::])/(n_sample-60)
Rh = sum(rho[60::])/(n_sample-60)

R = sum(res_img[60::])/(n_sample-60)
plt.imshow(R, cmap='gray')
plt.show()

print("bla")
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

# npix = 100
# slide = np.ones((npix, npix)) #make npix*npix image
#
# ncellmax = 10   #max no of cells
# cellrad = 9.5   #radius of cells
# stddev = 0.1 # noise standard deviation
#
# ncell = 5 + math.ceil((ncellmax - 5) * rd.rand()) #random number of cells at leas 5
# pbad = 0.25 + 0.5* rd.rand()        #probability a cell is bad
# A = npix * rd.rand(2, ncell)
# xycell = np.ceil(npix * rd.rand(2, ncell))
#
# for icell in range(0,ncell-1):
#     if rd.rand() < pbad:
#         slide = function.putbad(slide, xycell[0,icell], xycell[1,icell], cellrad)
#     else:
#         slide = function.putgood(slide, xycell[0,icell], xycell[1,icell], cellrad)
#
# x,y = slide.shape
# slide = slide + stddev * rd.rand(x,y)
#
# plt.imshow(slide,cmap='gray')
# plt.show()
#
#




