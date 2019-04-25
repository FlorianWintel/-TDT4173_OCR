#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import zipfile
import matplotlib.pyplot as plt
import string

from sklearn.feature_selection import VarianceThreshold, SelectPercentile ,chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def load_data(archiv_name = "dataset/chars74k-lite.zip"):
    """
    Load data from zip file, create label based on name of folder.
    :param archiv_name: name of zip file
    :returns: loaded data samples with shape [samples, features], targets with shape [samples]
    """
    assert zipfile.is_zipfile(archiv_name), "Is dataset in dataset/chars74k-lite.zip?"
    with zipfile.ZipFile(archiv_name, 'r') as zf:
        data=list()
        labels=list()
        for fname in zf.namelist():
            if fname[-4:] == ".jpg":
                with zf.open(fname) as file:
                    data.append(np.resize(plt.imread(file),20*20))
                    labels.append(string.ascii_lowercase.find(label))
            else:
                label = fname[-2:-1]
    print("Training dataset loaded.")
    return np.array(data), np.array(labels)

def init_transform(X,Y,switch, p=.9):
    """
    Perform initialization of data transformation on dataset.
    :param X: numpy array with shape [samples,features]
    :param Y: numpy array with shape [samples]
    :returns: list of tuples of transformation (tr1,tr2,tr3)
    """
    tr1 = MinMaxScaler()
    X_t=tr1.fit_transform(np.float64(X))
    
    if switch:
        tr2 = VarianceThreshold(threshold=(p * (1 - p)))
        X_t = tr2.fit_transform(X_t)
    else:
        tr2 = SelectPercentile(chi2, 100*p)
        X_t = tr2.fit_transform(X_t, Y)
        
    tr3 = PCA(np.uint8(0.5*X_t.shape[1]))
    tr3.fit(X_t)
    return (tr1,tr2,tr3)

def data_transform(X, transform):
    """
    Perform data transformation - normalization and feauture reduction
    :param X: numpy array with shape [samples,features]
    :param transform: list of tuples of transformation (tr1,tr2,tr3)
    :returns: transformed data numpy array with shape [samples,features]
    """
    # Normalization
    X_t = transform[0].transform(X)
    X_t = background_correction(X_t)
    # Feature reduction
    X_t = transform[1].transform(X_t)
    X_t = transform[2].transform(X_t)
    return X_t

def background_correction(X, p=.9):
    """
    Correction of background color. Leter is darker than background.
    :param X: numpy array with shape [samples,features]
    :param p: percent of edge pixels which value should be bigger than 124
    :returns: edited dataset
    """
    X_new = np.empty(X.shape)
    i=0
    for row in X:
        example = np.resize(row,(20,20))
        if sum(example[0][:] + example[-1][:] + example[1:-1][0] + example[1:-1][-1]) < p*(0.5*76):
            X_new[i] = 1 - row
        else:
            X_new[i] = row
        i += 1
    return X_new

def main():
    X,Y = load_data()
    tr1 = init_transform(X,Y,1)
    tr2 = init_transform(X,Y,0)
    X1 = data_transform(X,tr1)
    X2 = data_transform(X,tr2)
    print(X.shape,X1.shape,X2.shape)

if __name__== "__main__":
    main()    