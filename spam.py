#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:41:44 2020

@author: gumenghan
"""

import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
tag_train=[]
sentences_train=[]
with open('train.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        tag_train.append(1 if row[0]=='spam' else 0)
        item=row[1].split(' ')
        tmp=""
        for i in item:
            tmp=tmp+re.sub('[^a-zA-Z]','',i)
            
            tmp=tmp+" "
        sentences_train.append(tmp)

sentences_train=sentences_train[1:]
tag_train=tag_train[1:]

vectorizer = CountVectorizer()
#计算个词语出现的次数
X = vectorizer.fit_transform(sentences_train)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()

weight = X.toarray()

pca = PCA(n_components=2)
pca.fit(weight)

W=pca.transform(weight)

clf = GaussianNB()
clf.fit(W,tag_train)
#print ("==Predict result by predict==")
#print(clf.predict([[-0.8, -1]]))
#print ("==Predict result by predict_proba==")
#print(clf.predict_proba([[-0.8, -1]]))
#print ("==Predict result by predict_log_proba==")
#print(clf.predict_log_proba([[-0.8, -1]]))

indexes=[]
predictions=[]
raw_test_data=[]
with open('test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        indexes.append(row[0])
        item=row[1].split(' ')
        tmp=""
        for i in item:
            tmp=tmp+re.sub('[^a-zA-Z]','',i)
            
            tmp=tmp+" "
        raw_test_data.append(tmp)

indexes=indexes[1:]
raw_test_data=raw_test_data[1:]
vectorizer1 = CountVectorizer()
X1 = vectorizer1.fit_transform(raw_test_data)
#获取词袋中所有文本关键词
word1 = vectorizer1.get_feature_names()

weight1 = X1.toarray()
pca.fit(weight1)

W1=pca.transform(weight1)
for i in range(0,len(W1)):
    predictions.append('spam' if clf.predict([W1[i]])[0]==1 else 'ham')


res = pd.DataFrame({'SmsId':indexes,'Label':predictions})
res.to_csv("sampleSubmission.csv",index=False,sep=',')

