# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
from matplotlib import image
from scipy.cluster.vq import kmeans as km


def metric(img=None, chg_ref=None):

    chg_ref = np.array(chg_ref, dtype=np.float32)
    chg_ref = chg_ref / np.max(chg_ref)

    img = np.reshape(img, [-1])
    chg_ref = np.reshape(chg_ref, [-1])

    loc1 = np.where(chg_ref == 1)[0]
    num1 = np.sum(img[loc1] == 1)
    acc_chg = np.divide(float(num1), float(np.shape(loc1)[0]))

    loc2 = np.where(chg_ref == 0)[0]
    num2 = np.sum(img[loc2] == 0)
    acc_un = np.divide(float(num2), float(np.shape(loc2)[0]))

    acc_all = np.divide(float(num1 + num2), float(np.shape(loc1)[0] + np.shape(loc2)[0]))

    loc3 = np.where(img == 1)[0]
    num3 = np.sum(chg_ref[loc3] == 1)
    acc_tp = np.divide(float(num3), float(np.shape(loc3)[0]))

    print('')
    print('Accuracy of Unchanged Regions is: %.4f' % (acc_un))
    print('Accuracy of Changed Regions is:   %.4f' % (acc_chg))
    print('The True Positive ratio is:       %.4f' % (acc_tp))
    print('Accuracy of all testing sets is : %.4f' % (acc_all))

    return acc_un, acc_chg, acc_all, acc_tp


def getTrainSamples(index, im1, im2, number=4000):

    loc = np.where(index != 1)[0]
    perm = np.random.permutation(np.shape(loc)[0])

    ind = loc[perm[0:number]]

    return im1[ind, :], im2[ind, :]


def normlize(data):
    meanv = np.mean(data, axis=0)
    stdv = np.std(data, axis=0)

    delta = data - meanv
    data = delta / stdv

    return data


def linear_sfa(fcx, fcy, vp, shape):

    delta = np.matmul(fcx, vp) - np.matmul(fcy, vp)

    delta = delta / np.std(delta, axis=0)

    differ_map = delta  # utils.normlize(delta)

    delta = delta**2

    magnitude = np.sum(delta, axis=1)

    vv = magnitude / np.max(magnitude)

    im = np.reshape(kmeans(vv), shape[0:-1])

    return im, magnitude, differ_map


def data_loader(area=None):

    img1_path = area + '/img_t1.mat'
    img2_path = area + '/img_t2.mat'
    change_path = area + '/chg_ref.bmp'

    mat1 = sio.loadmat(img1_path)
    mat2 = sio.loadmat(img2_path)

    img1 = mat1['im']
    img2 = mat2['im']

    chg_map = image.imread(change_path)

    return img1, img2, chg_map


def kmeans(data):
    shape = np.shape(data)
    # print((data))
    ctr, _ = km(data, 2)

    for k1 in range(shape[0]):
        if abs(ctr[0] - data[k1]) >= abs(ctr[1] - data[k1]):
            data[k1] = 0
        else:
            data[k1] = 1
    return data
