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
    print("Load complete.")
    return np.array(data), np.array(labels)
    
def histogram_scale(X):
    """
    If histogram is narrowed, then is spread.
    :param X: numpy array with shape [samples,features]
    :returns: corrected dataset
    """
    min_max_scaler = MinMaxScaler()
    X_new = min_max_scaler.fit_transform(np.float64(X))
    return X_new, min_max_scaler

def background_correction(X, p=.9):
    """
    Correction of background color. Leter is darker than background.
    :param X: numpy array with shape [samples,features]
    :param p: percent of edge pixels which value should be bigger than 124
    :returns: corrected dataset
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

def feature_selection(X, Y, switch, p=.9):
    """
    Feature selection by two methods: VarianceThreshold,SelectPercentile. And then use PCA and reduce number on half
    :param X: numpy array with shape [samples,features]
    :param Y: numpy array with shape [samples]
    :param p: parameter for VarianceThreshold percent of samples to left the feature or for SelectPercentile
    :returns: 
    """
    if switch:
        sel = VarianceThreshold(threshold=(p * (1 - p)))
        X_new = sel.fit_transform(X)
    else:
        sel = SelectPercentile(chi2, p)
        X_new = sel.fit_transform(X, Y)
    pca = PCA(np.uint8(0.5*X_new.shape[1]))
    X_new = pca.fit(X_new).transform(X_new)
    return X_new, sel, pca

def main():
    X,Y = load_data()
    X,_ = histogram_scale(X)
    X_b = background_correction(X)
    X1,_ = feature_selection(X_b,Y,1)
    X2,_ = feature_selection(X,Y,0)
    print(X.shape,X1.shape,X2.shape)
    for i in range(20):
        example=np.resize(X[i],(20,20))
        print(sum(example[0][:] + example[-1][:] + example[1:-1][0] + example[1:-1][-1]),.9*(0.5*76))
        plt.imshow(np.resize(X_b[i],(20,20)),cmap='Greys')
        plt.show()

if __name__== "__main__":
    main()    