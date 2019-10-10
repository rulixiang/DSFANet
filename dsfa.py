# -*- coding: utf-8 -*-
import argparse
import logging
import os

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

import utils
from model import dsfa

net_shape = [128, 128, 6] 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

def parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e','--epoch',help='epoches',default=2000, type=int)
    parser.add_argument('-l','--lr',help='learning rate',default=5*1e-4, type=float)
    parser.add_argument('-r','--reg',help='regularization parameter',default=1e-4, type=float)
    parser.add_argument('-t','--trn',help='number of training samples',default=3000, type=int)
    parser.add_argument("-i",'--iter',  help="max iteration", default=10, type=int)
    parser.add_argument('-g','--gpu', help='GPU ID', default='0')
    parser.add_argument('--area',help='datasets', default='river')
    args = parser.parse_args()

    return args

def main(img1, img2, chg_map, args=None):

    img_shape = np.shape(img1)

    im1 = np.reshape(img1, newshape=[-1,img_shape[-1]])
    im2 = np.reshape(img2, newshape=[-1,img_shape[-1]])

    im1 = utils.normlize(im1)
    im2 = utils.normlize(im2)

    chg_ref = np.reshape(chg_map, newshape=[-1])

    imm = None
    all_magnitude = None
    differ = np.zeros(shape=[np.shape(chg_ref)[0],net_shape[-1], args.iter])

    # load cva pre-detection result
    ind = sio.loadmat(args.area+'/cva_ref.mat')
    cva_ind = ind['cva_ref']
    cva_ind = np.reshape(cva_ind, newshape=[-1])

    for k1 in range(args.iter):

        logging.info('In %2d-th iteration········' % (k1))

        i1, i2 = utils.getTrainSamples(cva_ind, im1, im2, args.trn)

        loss_log, vpro, fcx, fcy, bval = dsfa(
            xtrain=i1, ytrain=i2, xtest=im1, ytest=im2, net_shape=net_shape, args=args)

        imm, magnitude, differ_map = utils.linear_sfa(fcx, fcy, vpro, shape=img_shape)

        magnitude = np.reshape(magnitude, img_shape[0:-1])
        differ[:, :, k1] = differ_map

        if all_magnitude is None:
            all_magnitude = magnitude / np.max(magnitude)
        else:
            all_magnitude = all_magnitude + magnitude / np.max(magnitude)


    change_map = np.reshape(utils.kmeans(np.reshape(all_magnitude, [-1])), img_shape[0:-1])

    logging.info('Max value of change magnitude: %.4f'%(np.max(all_magnitude)))
    logging.info('Min value of change magnitude: %.4f'%(np.min(all_magnitude)))

    # magnitude
    acc_un, acc_chg, acc_all2, acc_tp = utils.metric(1-change_map, chg_map)
    acc_un, acc_chg, acc_all3, acc_tp = utils.metric(change_map, chg_map)
    plt.imsave('results.png',all_magnitude, cmap='gray')
    #plt.show()

    return None


if __name__ == '__main__':
    args = parser()
    img1, img2, chg_map = utils.data_loader(area=args.area)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(img1, img2, chg_map, args=args)
