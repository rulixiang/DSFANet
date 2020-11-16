import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import io as sio
from sklearn import preprocessing
from sklearn.cluster import KMeans


def normalize(data):
    meanv = np.mean(data, axis=0)
    stdv = np.std(data, axis=0)

    delta = data - meanv
    data = delta / stdv

    return data

def load_dataset(norm_flag=True):

    imgX = sio.loadmat('river/river_before.mat')['river_before']
    imgY = sio.loadmat('river/river_after.mat')['river_after']

    imgX = np.reshape(imgX, newshape=[-1, imgX.shape[-1]])
    imgY = np.reshape(imgY, newshape=[-1, imgY.shape[-1]])

    GT = sio.loadmat('river/groundtruth.mat')['lakelabel_v1']

    if norm_flag:
        X = preprocessing.StandardScaler().fit_transform(imgX)
        Y = preprocessing.StandardScaler().fit_transform(imgY)
        #X = normalize(imgX)
        #Y = normalize(imgY)

    return X, Y, GT

def cva(X, Y):

    diff = X - Y
    diff_s = (diff**2).sum(axis=-1)

    return np.sqrt(diff_s)

def SFA(X, Y):
    '''
    see http://sigma.whu.edu.cn/data/res/files/SFACode.zip
    '''
    norm_flag = True
    m, n = np.shape(X)
    meanX = np.mean(X, axis=0)
    meanY = np.mean(Y, axis=0)

    stdX = np.std(X, axis=0)
    stdY = np.std(Y, axis=0)

    Xc = (X - meanX) / stdX
    Yc = (Y - meanY) / stdY

    Xc = Xc.T
    Yc = Yc.T

    A = np.matmul((Xc-Yc), (Xc-Yc).T)/m
    B = (np.matmul(Yc, Yc.T)+np.matmul(Yc, Yc.T))/2/m

    D, V = scipy.linalg.eig(A, B)  # V is column wise
    D = D.real
    #idx = D.argsort()
    #D = D[idx]

    if norm_flag is True:
        aux1 = np.matmul(np.matmul(V.T, B), V)
        aux2 = 1/np.sqrt(np.diag(aux1))
        V = V * aux2
    #V = V[:,0:3]
    X_trans = np.matmul(V.T, Xc).T
    Y_trans = np.matmul(V.T, Yc).T

    return X_trans, Y_trans

