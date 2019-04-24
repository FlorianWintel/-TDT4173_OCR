#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import zipfile
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

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
        targets = "abcdefghijklmnopqrstuvwxyz"
        for fname in zf.namelist():
            if fname[-4:] == ".jpg":
                with zf.open(fname) as file:
                    data.append(np.resize(plt.imread(file),20*20))
                    labels.append(targets.find(label))
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
    min_max_scaler = MinMaxScaler(feature_range=(0, 255))
    X_new = min_max_scaler.fit_transform(X)
    return X_new

def background_correction(X, p=.7):
    """
    Correction of background color. Leter is brighter than background.
    :param X: numpy array with shape [samples,features]
    :param p: percent of edge pixels which value should be bigger than 124
    :returns: corrected dataset
    """
    X_new = np.empty(X.shape)
    i=0
    for row in X:
        example = np.resize(row,(20,20))
        if sum(example[0][:] + example[-1][:] + example[1:-1][0] + example[1:-1][-1]) > p*(124*76):
            X_new[i] = np.uint8(255 - row)
        else:
            X_new[i] = row
        i += 1
    return X_new

def feature_selection(X, Y, switch, p=.001, k=200):
    """
    Correction of background color. Leter is brighter than background.
    :param X: numpy array with shape [samples,features]
    :param Y: numpy array with shape [samples]
    :param p: parameter for variance threshold percent of samples to left the feature
    :param k: parameter for SelectKBest number of returned features
    :returns: 
    """
    if switch:
        sel = VarianceThreshold(threshold=(p * (1 - p)))
        X_new = sel.fit_transform(X)
    else:
        sel = SelectKBest(chi2, k)
        X_new = sel.fit_transform(X, Y)
    return X_new

def main():
    X,Y = load_data()
    X = histogram_scale(X)
    X = background_correction(X)
    X1 = feature_selection(X,Y,1)
    X2 = feature_selection(X,Y,0)
    print(X.shape,X1.shape,X2.shape)

if __name__== "__main__":
    main()    