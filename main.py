import numpy as np
import pylab as pl
from PIL import Image, ImageOps
from numpy import array
import scipy
from scipy import signal
from numpy.fft import fft2, ifft2, fftshift, fft, ifft, irfft2, rfft2, hfft,  ifftshift
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
import sympy as sy
import numpy.random as rd
import numpy.linalg as lin
from scipy.fftpack import convolve
import function
import torch
from torch import nn
#
import scipy.special as sps

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


#######################################################


test_mat = abs(np.around(rd.normal(0,1,(10,10)) * 10))
conv = abs(np.around(rd.normal(0,1,(2,2)) * 10))
conv_mat = conv/ sum(sum(conv))
conv_mat_pad = np.pad(conv_mat, (0, len(test_mat)- len(conv_mat)) )
def direct_linear_2D(x,h):
    r = np.zeros((len(x),len(x)))

    for k in range(len(x) ):
        for l in range(len(x)):

            for m in range(len(x)):
                for n in range(len(x)):
                    if k-m >= 0 and (l-n)>= 0 and (k-m) < len(h) and (l-n) < len(h) :
                        r[k,l] = r[k,l] + (x[m,n] * h[(k-m),(l-n)])
    return r

direct_lin_conv_mat = direct_linear_2D(test_mat, conv_mat)
scy_conv_mat = scipy.signal.convolve(test_mat, conv_mat, method = 'direct')[0:len(test_mat), 0:len(test_mat)]
#do direct linear convolution through fft2 and zero padding
final_lin_conv_mat_fft = ifft2(fft2(conv_mat, (2*len(test_mat),2*len(test_mat))) * fft2(test_mat,(2*len(test_mat),2*len(test_mat)))).real[0:len(test_mat), 0:len(test_mat)]

print(np.allclose(direct_lin_conv_mat,final_lin_conv_mat_fft, atol= 1e-9))

#2D circular convolution
#matrices have to be same size and conv_matrix has to be zero padded
def direct_circular_2D(x,h):
    r = np.zeros((len(x),len(x)))
    for k in range(len(x)):
        for l in range(len(x)):

            for m in range(len(x)):
                for n in range(len(x)):
                    #print('k-p ' + str((k - p)))
                    #print((k - p) % len(g))
                    r[k,l] = r[k,l] + (x[m,n] * h[ (k-m) % (len(h)) , (l-n) % (len(h)) ])

    return r


direct_circ_conv_mat = direct_circular_2D(test_mat, conv_mat_pad)
final_circ_mat_fft = ifft2(fft2(conv_mat, (10,10)) * fft2(test_mat) ).real
print(np.allclose(direct_circ_conv_mat,final_circ_mat_fft, atol= 1e-9))

def transpose_conv_2d(X,H, padding):
#H is kernel
#X and H have to be squared
#padding reduces the size of the output equally
    X_pad = np.pad(X, (0,padding))
    for x in range(padding):
        X_pad[len(X)+x,:] = X_pad[x,:]
        X_pad[:, len(X) + x] = X_pad[:,x]
        X_pad[len(X) + x, len(X) + x] =  X_pad[x,x]
    output_size = len(X_pad) - len(H)
    Z = np.zeros((output_size,output_size))
    #pad X

#stride
    k = 0

    for i in range(output_size):
        l = 0
        for j in range(output_size):
            Z[k,l] =  sum(sum(X_pad[i: (i+len(H)),j: (j+len(H)) ] * H))
            l = l + 1
        k = k + 1

    return Z

trans_circ_mat = transpose_conv_2d(test_mat, conv, len(conv))
final_trans_circ_mat_fft = ifft2(  fft2(conv, (10,10)).conj() * fft2(test_mat) ).real
print(np.allclose(trans_circ_mat,final_trans_circ_mat_fft, atol= 1e-9))
# ################################################################

#norm parseval theorem


n = 100
x = np.random.rand(n)
f = fft(x)

print(sum(abs(x)**2))

print(sum(abs(f)**2)/n)
print(np.linalg.norm(x)**2)
print(np.linalg.norm(f)**2/n)


#####################################
gray_img = mpimg.imread('jupiter1.tif')


# get psf from satellite
org_img = array(gray_img)#/sum(sum(np.array(gray_img)))
four_conv = fft2(org_img)#, axes = (1,0) ) #/ np.sqrt(256**2)
# plt.imshow(org_img, cmap='gray')
# plt.show()



xpos = 234
ypos = 85  # Pixel at centre of satellite
sat_img_org = org_img[ypos - 16: ypos + 16, xpos - 16:xpos + 16]
sat_img = sat_img_org / (sum(sum(sat_img_org)))
sat_img[sat_img < 0.05*np.max(sat_img)] = 0

# plt.imshow(sat_img, cmap='gray')
# plt.show()


pad_img = np.zeros((256,256))
pad_img[0:32,0:32] = sat_img
fourier_img = fft2(sat_img, (256,256))#, axes =(1,0) )
fourier_img2 = fft2(pad_img)
# # naive inversion by direct division
print(np.allclose(fourier_img, fourier_img2, atol= 1e-10))





lam = 0.100#19

tikh_img_store = ( ifft2( four_conv * np.conj(fourier_img) / (lam ** 2 +  abs(fourier_img)**2 )))
tikh_img = tikh_img_store.real#[0:256,0:256]

Z_fft = ifft2(four_conv * np.conj(fourier_img)).real
Z =  transpose_conv_2d(org_img, sat_img, len(sat_img))
print(np.allclose(Z, Z_fft, atol= 1e-9))


alpha = 0.0056724
L_org = np.array([[ 0, -1, 0],[ -1, 4, -1],[ 0, -1, 0]])
four_L = fft2(  L_org, (256,256))


print('bla')


# # alphas = np.linspace(0.000001,0.1,1000)
# # #alphas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) * 1e-3
# alphas = np.logspace(-9,1, 200)
# norm_f = [None] * len(alphas)
# norm_data = [None] * len(alphas)
#
# for i in range(0, len(alphas)):
#
#     #tikh_img = four_conv * np.conj(fourier_img)/ (alphas[i] ** 2 +  abs(fourier_img)**2)
#     reg_img = four_conv * np.conj(fourier_img)/ (alphas[i] * abs(four_L) +  abs(fourier_img)**2)
#     x = np. matrix.flatten(reg_img)
#     norm_f[i] = np.sqrt( sum(sum( abs(reg_img.conj() * four_L * reg_img))))/256
#     norm_data[i] = np.linalg.norm(ifft2(four_conv - reg_img * fourier_img))
# #   norm_data[i] = np.linalg.norm(four_conv - reg_img * fourier_img)/256
# reg_img = four_conv * np.conj(fourier_img) / (alphas[109] * abs(four_L) + abs(fourier_img) ** 2)
# c = ifft2(reg_img).real
# plt.imshow(c, cmap='gray')
# plt.show()
#
#
# fig2 = plt.figure()
# ax = fig2.add_subplot()
# ax.set_xscale('log')
# ax.set_yscale('log')
# # plt.xlim((10,1e4))
# # plt.ylim((1e3,1e6))
# # print(lambas[::100])
#
# plt.scatter(norm_data, norm_f)
# k = 0
# for i, txt in enumerate(alphas[::5]):
#
# #i = 50
# #txt = alphas[i]
#     ax.annotate(np.around(txt,9), (norm_data[k], norm_f[k]))
#     k = k + 5
# # i = 109
# # txt = alphas[i]
# # ax.annotate(np.around(txt,5), (norm_data[i], norm_f[i]))
# plt.show()



#sample from prior
number = 25
siz = int(np.sqrt(number))

L = np.zeros((number,number))

l = np.array([[1,-1],[-1,1]])

l_boundud = np.zeros((number - siz + 1, number - siz + 1))
l_boundud[0, 0], l_boundud[0, -1], l_boundud[-1, 0], l_boundud[-1, -1] = 1, -1, -1, 1

l_lr = np.zeros((siz,siz))
l_lr[0,0],l_lr[0,-1],l_lr[-1,0],l_lr[-1,-1] = 1,-1,-1,1

l_ud = np.zeros((siz + 1, siz + 1))
l_ud[0,0],l_ud[0,-1],l_ud[-1,0],l_ud[-1,-1] = 1,-1,-1,1



#sample the shit out of L
n_sample = 1
samples =  np.zeros((n_sample,siz,siz))

for n in range(0, n_sample):
    v_2 = np.zeros((siz, siz))
    #top to bottom first row
    for i in range(0,siz):
        L[i:i+len(l_boundud), i:i + len(l_boundud)] = L[i:i + len(l_boundud), i:i + len(l_boundud)] + l_boundud
        rand_num = (1 / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
        # rand_num = np.array([-1, 1]) * rd.normal(0, 1)
        v_2[0,i], v_2[-1,i] = [v_2[0,i], v_2[-1,i]] + np.array(rand_num)

    #all normal consecutive neighbours
    x = np.arange(0,number+siz,siz)
    for j in range(1,len(x)):
        for i in range(x[j-1], x[j]-1):
            L[i:i + len(l), i:i + len(l)] = L[i:i + len(l), i:i + len(l)]  + l

    for j in range(0, siz):
        for i in range(0, siz-1):
            rand_num = (1 / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
            # rand_num = np.array([-1, 1]) * rd.normal(0, 1)
            v_2[j, i], v_2[j, i+1] = [v_2[j, i], v_2[j, i+1]] + np.array(rand_num)

    #all normal up and down neighbours

    for j in range(1,len(x)-1):
        for i in range(x[j-1], x[j]):
            L[i:i + len(l_ud), i:i + len(l_ud)] = L[i:i + len(l_ud), i:i + len(l_ud)] + l_ud

    for j in range(0, siz-1):
        for i in range(0, siz):
            rand_num = (1 / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
            # rand_num = np.array([-1, 1]) * rd.normal(0, 1)
            v_2[j, i], v_2[j+1, i] = [v_2[j, i], v_2[j+1, i]] + np.array(rand_num)

    #all left right boundaries neighbours

    for i in range(0, siz):
        print(i)
        L[x[i]:x[i] + len(l_lr), x[i]:x[i] + len(l_lr)] = L[x[i]:x[i] + len(l_lr), x[i]:x[i] + len(l_lr)] + l_lr
        rand_num = (1 / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
        # rand_num = np.array([-1, 1]) * rd.normal(0, 1)
        v_2[i, 0], v_2[i, -1] = [v_2[i, 0], v_2[i, -1]] + np.array(rand_num)


    samples[n] = v_2

L = np.zeros((number,number))
v_2 = np.zeros((siz, siz))
k = 0
#top to bottom first row
for i in range(0,siz):
    k = k + 1
    L[i:i+len(l_boundud), i:i + len(l_boundud)] = L[i:i + len(l_boundud), i:i + len(l_boundud)] + l_boundud
    rand_num = np.sqrt(2) * np.array([-1, 1])
    # rand_num = np.array([-1, 1]) * rd.normal(0, 1)
    v_2[0,i], v_2[-1,i] = [v_2[0,i], v_2[-1,i]] + np.array(rand_num)

#all normal consecutive neighbours
x = np.arange(0,number+siz,siz)
for j in range(1,len(x)):
    for i in range(x[j-1], x[j]-1):
        k = k + 1
        L[i:i + len(l), i:i + len(l)] = L[i:i + len(l), i:i + len(l)]  + l

for j in range(0, siz):
    for i in range(0, siz-1):
        rand_num = np.sqrt(2) * np.array([-1, 1])
        # rand_num = np.array([-1, 1]) * rd.normal(0, 1)
        v_2[j, i], v_2[j, i+1] = [v_2[j, i], v_2[j, i+1]] + np.array(rand_num)

#all normal up and down neighbours

for j in range(1,len(x)-1):
    for i in range(x[j-1], x[j]):
        k = k + 1
        L[i:i + len(l_ud), i:i + len(l_ud)] = L[i:i + len(l_ud), i:i + len(l_ud)] + l_ud

for j in range(0, siz-1):
    for i in range(0, siz):
        rand_num = np.sqrt(2) * np.array([-1, 1])
        # rand_num = np.array([-1, 1]) * rd.normal(0, 1)
        v_2[j, i], v_2[j+1, i] = [v_2[j, i], v_2[j+1, i]] + np.array(rand_num)

#all left right boundaries neighbours

for i in range(0, siz):
    k = k + 1
    L[x[i]:x[i] + len(l_lr), x[i]:x[i] + len(l_lr)] = L[x[i]:x[i] + len(l_lr), x[i]:x[i] + len(l_lr)] + l_lr
    rand_num = np.sqrt(2) * np.array([-1, 1])
    # rand_num = np.array([-1, 1]) * rd.normal(0, 1)
    v_2[i, 0], v_2[i, -1] = [v_2[i, 0], v_2[i, -1]] + np.array(rand_num)





v, w = np.linalg.eig(L)
u1, v1, vh1 = np.linalg.svd(L)

W = np.ndarray(shape = (number,number), dtype = 'csingle')
W_trans = np.zeros((number,number))
for m in range(number):
    for n in range(number):
        W[m,n] =  np.exp( -2j * np.pi * m * n / number)/ np.sqrt(number)

W_trans =   W.T.conj() / np.sqrt(number)




LTL = np.matmul(L.T,L)
LLT = np.matmul(L,L.T)
print(np.allclose(LTL,LLT))


SIGMA =(np.matmul(W_trans, np.matmul(L,W))).real

lW = np.matmul(L,W).real
lW2 = np.matmul(W,L).real



WD =np.matmul(W, 8 * np.identity(number) ).real


number = 2
W = np.ndarray(shape = (number,number), dtype = 'csingle')
W_trans = np.zeros((number,number))
for m in range(number):
    for n in range(number):
        W[m,n] =  np.exp( -2j * np.pi * m * n / number) / np.sqrt(number)

W_trans =   W.T.conj() / np.sqrt(number)







N = 2
R = np.ndarray(shape = (2,2), dtype = 'csingle')
for i in range(len(l)):
    for k in range(len(l)):
        R[i,k] =  np.exp(-2j * np.pi * i * k / N) / np.sqrt(N)



LR =  np.matmul(l, R).real

RD = np.matmul(R,[[0,0],[0,2]]).real

RTLR = (np.matmul(R.T.conj() , np.matmul(l,R))).real
a_gamma = 1
a_rho = 1
b_gamma = 1e-4
b_rho = 1e-4
m= 256**2

n_sample = 70
res_img = np.zeros((n_sample,256,256))
rho =  np.zeros(n_sample)
gamma =  np.zeros(n_sample)


#intialize first sample

rho[0] =  5.16e-5

gamma[0] = 0.218

L_i = 16
func_f = sum( sum(  abs(four_conv)**2 * rho[0]/ gamma[0] *L_i /abs(fourier_img)**2 / ( 1+ rho[0]/ gamma[0] * L_i/abs(fourier_img)**2)))/256**2

func_g = sum(sum(abs(fourier_img)**2 )) + sum(sum( 1+ rho[0]/ gamma[0] * L_i/abs(fourier_img)**2) )

number = 256**2
siz = int(np.sqrt(number))



v_2 = np.zeros((siz, siz))
#top to bottom first row
for i in range(0,siz):
    # rand_num = (np.sqrt(rho[0]) / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
    rand_num =  np.array([-1, 1]) * rd.normal(0, 1)
    v_2[0,i], v_2[-1,i] = [v_2[0,i], v_2[-1,i]] + np.array(rand_num)

#all normal consecutive neighbours
x = np.arange(0,number+siz,siz)
for j in range(0, siz):
    for i in range(0, siz-1):
        #rand_num = (np.sqrt(rho[0]) / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
        rand_num =  np.array([-1, 1]) * rd.normal(0, 1)
        v_2[j, i], v_2[j, i+1] = [v_2[j, i], v_2[j, i+1]] + np.array(rand_num)

#all normal up and down neighbours
for j in range(0, siz-1):
    for i in range(0, siz):
        # rand_num = (np.sqrt(rho[0]) / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
        rand_num =np.array([-1, 1]) * rd.normal(0, 1)
        v_2[j, i], v_2[j+1, i] = [v_2[j, i], v_2[j+1, i]] + np.array(rand_num)

#all left right boundaries neighbours
for i in range(0, len(x)-1):
    # rand_num = (np.sqrt(rho[0]) / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
    rand_num = np.array([-1, 1]) * rd.normal(0, 1)
    v_2[i, 0], v_2[i, -1] = [v_2[i, 0], v_2[i, -1]] + np.array(rand_num)

v_rd = rd.normal(0,1,256**2).reshape((256,256))

w = np.sqrt(gamma[0]) * np.conj(fourier_img) * fft2(v_rd) + np.sqrt(rho[0]) * fft2(v_2)

img_store =  (gamma[0] * four_conv * np.conj(fourier_img) + w)/ ( rho[0] * abs(four_L) +   gamma[0] * abs(fourier_img)**2)

im = ifft2( img_store ).real
res_img[0]= im
#img = org_img + np.reshape(rd.normal(0, 1, 256**2),(256,256))

plt.imshow(res_img[0], cmap='gray')
plt.show()
norm_res = np.linalg.norm( img_store * fourier_img - fft2(org_img))/256
print(norm_res)

#res_img[0][res_img[0]<0] = 0
norm_L = np.sqrt(sum(sum(abs(img_store.conj() * four_L * img_store)))) / 256
print(norm_L)

shape_gamma, scale_gamma = m/2 + a_gamma, 1/(0.5 * norm_res**2 + b_gamma)
shape_rho, scale_rho = m/2 + a_rho, 1/(0.5 * norm_L**2 + b_rho)
mean_gamma = shape_gamma*scale_gamma
mean_rho = shape_rho * scale_rho

for n in range(1, n_sample):

    norm_L = np.sqrt(sum(sum(abs(img_store.conj() * four_L * img_store))))/256
    norm_res = np.linalg.norm( img_store * fourier_img - fft2(org_img))/256
    shape_gamma, scale_gamma = m/2 + a_gamma, 1/(0.5 * norm_res**2 + b_gamma)#1,1e4 #

    gamma[n] =np.random.default_rng().gamma(shape_gamma, scale_gamma)
    shape_rho, scale_rho = m/2 + a_rho, 1/(0.5 *  norm_L**2 + b_rho)#1,1e4 #
    rho[n] =  np.random.default_rng().gamma(shape_rho, scale_rho)
    v_2 = np.zeros((siz, siz))
    # top to bottom first row
    for i in range(0, siz):
        # rand_num = (np.sqrt(rho[0]) / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
        rand_num = np.array([-1, 1]) * rd.normal(0, 1)
        v_2[0, i], v_2[-1, i] = [v_2[0, i], v_2[-1, i]] + np.array(rand_num)

    # all normal consecutive neighbours
    x = np.arange(0, number + siz, siz)
    for j in range(0, siz):
        for i in range(0, siz - 1):
            # rand_num = (np.sqrt(rho[0]) / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
            rand_num = np.array([-1, 1]) * rd.normal(0, 1)
            v_2[j, i], v_2[j, i + 1] = [v_2[j, i], v_2[j, i + 1]] + np.array(rand_num)

    # all normal up and down neighbours
    for j in range(0, siz - 1):
        for i in range(0, siz):
            # rand_num = (np.sqrt(rho[0]) / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
            rand_num = np.array([-1, 1]) * rd.normal(0, 1)
            v_2[j, i], v_2[j + 1, i] = [v_2[j, i], v_2[j + 1, i]] + np.array(rand_num)

    # all left right boundaries neighbours
    for i in range(0, len(x) - 1):
        # rand_num = (np.sqrt(rho[0]) / np.sqrt(2)) * np.array([-1, 1]) * rd.normal(0, np.sqrt(2))
        rand_num = np.array([-1, 1]) * rd.normal(0, 1)
        v_2[i, 0], v_2[i, -1] = [v_2[i, 0], v_2[i, -1]] + np.array(rand_num)

    v_rd = rd.normal(0,1,256**2).reshape((256,256))
    w = np.sqrt(gamma[n]) * np.conj(fourier_img) * fft2(v_rd) + np.sqrt(rho[n]) * fft2(v_2)
    print(rho[n])
    print(gamma[n])
    img_store = (gamma[n] * four_conv * np.conj(fourier_img) + w) / (
                rho[n] * abs(four_L) + gamma[n] * abs(fourier_img) ** 2)
    res_img[n] = ifft2( (gamma[n] * four_conv * np.conj(fourier_img) + w )/ ( rho[n] * abs(four_L) +   gamma[n] * abs(fourier_img) ** 2) ).real

    # plt.imshow(res_img[n], cmap='gray')
    # plt.show()
G = sum(gamma[60::])/(n_sample-60)
Rh = sum(rho[60::])/(n_sample-60)

R = sum(res_img[60::])/(n_sample-60)
plt.imshow(R, cmap='gray')
plt.show()
plt.imshow(res_img[-2], cmap='gray')
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




