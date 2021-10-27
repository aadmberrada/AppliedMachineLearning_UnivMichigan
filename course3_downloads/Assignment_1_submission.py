#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 01:57:31 2021

@author: Abdoul_Aziz_Berrada
"""



import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()

cancer.DESCR

cancer.keys()


def answer_zero():
    
    return len(cancer['feature_names'])
answer_zero()


d=cancer["data"]
df=cancer["feature_names"]

cancerdf=pd.DataFrame(cancer.data, columns=["feature_names"])


cancer['feature_names']


len(cancer.target)

len(cancer.feature_names)

cancer_columns=np.append(cancer.feature_names, ["target"])

cancer_data=np.c_[cancer.data, cancer.target]

cancerdf=pd.DataFrame(cancer_data, columns=cancer_columns)


def answer_one():
    cancer_data=np.c_[cancer.target, cancer.data]
    cancer_columns=np.append(["target"],cancer.feature_names )
    return pd.DataFrame(cancer_data, columns=cancer_columns)

answer_one()


#
def answer_two():
    cancerdf = answer_one()
    counts = cancerdf.target.value_counts(ascending=True)
    counts.index = "malignant benign".split()
    return counts
counts=answer_two()
#




def answer_three():
    cancerdf = answer_one()
    X= cancerdf[cancer.feature_names]
    y=cancerdf["target"]
    return X, y
X,y=answer_three()



from sklearn.model_selection import train_test_split
def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=426, test_size=143)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test=answer_four()



from sklearn.neighbors import KNeighborsClassifier
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn=KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    return knn
knn=answer_five()



def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn=answer_five()
    return knn.predict(means)
pred_means=answer_six()


def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.predict(X_test)
prediction=answer_seven()


def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.score(X_test, y_test)
score=answer_eight()

