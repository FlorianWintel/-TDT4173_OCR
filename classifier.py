#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import load_data, init_transform, data_transform


def plot_confusion_matrix(y_true, y_pred, cm, normalize=False, title=None, cmap=plt.cm.Blues):
    classes = string.ascii_lowercase
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.show()
    return
    
    
def get_correct_and_cm(y_p, y_t):
    """
    Calculates the ratio of correct classifications and the confusion matrix for the given predictions.
    :param y_p: float array, predictions
    :param y_t: float array, target
    :returns: 
    """
    right = 0;
    for i in range(len(y_t)):
        if y_t[i] == y_p[i]:
            right += 1
    cm = confusion_matrix(y_t, y_p)
    return right/len(y_t), cm
 
"""
    
def init_knn():
    
    
    # k-NN
    n_nei = 8
    neigh = KNeighborsClassifier(n_neighbors=n_nei)
    neigh.fit(X_train, y_train)
    kNN_y_pred = neigh.predict(X_test)
    
    kNN_right, kNN_con = correct_class(kNN_y_pred, y_test)
    print ('kNN')
    print('correct classified: ' + str(kNN_right))
    #print('confusion matrix: \n' + str(kNN_con))
    plot_confusion_matrix(y_test, kNN_y_pred, classes=alphabet, cm=kNN_con, title='Confusion matrix, kNN')
    plt.show()
    return neigh

"""


def classify(crop, clf):
    """
    Applies classifier.
    :param crop: PIL image, a slice of the input image
    :param clf: list (tranformation, classifier)
    :returns: predicted class as string, prediction score as a float
    """
    # Apply data tranformation
    X = np.resize(np.array(crop),(1,20*20))
    X = data_transform(X,clf[0])

    prediction = string.ascii_lowercase[clf[1].predict(X)[0]]
    predict_score = np.max(clf[1].predict_proba(X))
    
    return prediction, predict_score
    

def predict(image, boxes, window_size, clf):
    """
    Predictor, calls the classifier on each found letter.
    :param image: input image as PIL image object
    :param boxes: list of tuples (x_min <int>, y_min <int>, detection score <float>)
    :param window_size: size of the sliding window, tuple of integers (x,y)
    :param clf: list [tranformation, classifier]
    :returns: list of tuples (x_min <int>, y_min <int>, class prediction <string>, prediction score <float>)
    """
    classified_boxes = []
    for box in boxes:
        x,y = box[0], box[1]
        prediction, predict_score = classify(image.crop((x,y,x+window_size[0],y+window_size[1])),clf)
        classified_boxes.append((x, y, prediction, predict_score))
    return classified_boxes


def init_clf():
    """
    Initializacion of classifier and data transformation.
    :returns: list of tuples (data transformation, classifier)
    """
    X,Y = load_data()
    tr = init_transform(X,Y,1)
    X = data_transform(X,tr)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    
    # SVM
    clf_svm = svm.SVC(gamma='scale', probability=True, C=10)
    clf_svm.fit(X_train, y_train)
    SVM_y_pred = clf_svm.predict(X_test)
    
    SVM_right, SVM_cm = get_correct_and_cm(SVM_y_pred,y_test)
    print ('Classifying with SVM')
    print('Correctly classified: %.2f' %(SVM_right*100))
    
    plot_confusion_matrix(y_test, SVM_y_pred, SVM_cm) 
    
    return (tr, clf_svm)

def main():
    init_clf()

if __name__== "__main__":
    main()