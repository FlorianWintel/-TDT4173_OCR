#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

def SVM(C=1000, gamma='scale'):
    """
    Initialization of KNN classifier.
    :param C: penalty parameter C of the error term
    :param gamma: kernel coefficient 
    :returns: classifier
    """
    clf_svm = svm.SVC(C=C, gamma=gamma, probability=True)
    
    return clf_svm

def KNN(n_nei=4):
    """
    Initialization of KNN classifier.
    :param n_nei: number of neighbors
    :returns: classifier
    """
    neigh = KNeighborsClassifier(n_neighbors=n_nei)

    return neigh

def RandomForest_with_AdaBoost():
    """
    Initialization of RandomForest with AdaBoost classifier.
    :returns: classifier
    """
    clf_rft = AdaBoostClassifier(RandomForestClassifier(n_estimators=200, max_depth=4))
    
    return clf_rft

def Decision_tree_with_AdaBoost():
    """
    Initialization of Decision tree with AdaBoost classifier.
    :returns: classifier
    """
    dtc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), algorithm="SAMME", n_estimators=200)
    
    return dtc

def init_clf(switch=0):
    """
    Initialization of classifier and data transformation.
    :returns: list of tuples (data transformation, classifier)
    """
    if switch==0:
        name = "SVM"
        clf = SVM()
    elif switch==1:
        name = "KNN"
        clf = KNN()
    elif switch==2:
        name = "RandomForest with AdaBoost"
        clf = RandomForest_with_AdaBoost()
    elif switch==3:
        name = "Decision tree with AdaBoost"
        clf = Decision_tree_with_AdaBoost()
    else:
        assert False, "No so many classifiers"
        
    X,Y = load_data()
    tr = init_transform(X,Y,1)
    X = data_transform(X,tr)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    
    print('Training the %s classifier ' %name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    right, cm = get_correct_and_cm(y_pred,y_test)
    print('Classifying with %s' %name)
    print('Correctly classified: %.2f%%' %(right*100))
    
    plot_confusion_matrix(y_test, y_pred, cm) 
    
    return (tr, clf)

def main():
    for i in range(4):
        init_clf(i)

if __name__== "__main__":
    main()