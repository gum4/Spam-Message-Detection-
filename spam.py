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
import torch
import torch.nn.functional as F
from torch.autograd import Variable








tag_train=[]
sentences_train=[]
with open('train.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        tag_train.append([1] if row[0]=='spam' else [0])
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

pca = PCA(n_components=10)
pca.fit(weight)

W=pca.transform(weight)

WW=torch.from_numpy(W).float()

TT=torch.from_numpy(np.array(tag_train)).float()

WW,TT=Variable(WW),Variable (TT)






#clf = GaussianNB()
#clf.fit(W,tag_train)
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
WW1=torch.from_numpy(W1).float()

#for i in range(0,len(W1)):
#    predictions.append('spam' if clf.predict([W1[i]])[0]==1 else 'ham')


#res = pd.DataFrame({'SmsId':indexes,'Label':predictions})
#res.to_csv("sampleSubmission.csv",index=False,sep=',')





#####
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 5572, 10, 10, 1

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)


# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(WW)

    # Compute and print loss
    loss = criterion(y_pred, TT)
    #if t % 100 == 99:
    #    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

model(WW1)

