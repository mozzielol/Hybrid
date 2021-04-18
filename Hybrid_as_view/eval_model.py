import torch
import sys
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import importlib.util


def next_batch(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        X_batch = torch.tensor(X[i: i+batch_size]) / 255.
        y_batch = torch.tensor(y[i: i+batch_size])
        yield X_batch.to('cuda'), y_batch.to('cuda')
        
def _load_stl10(prefix="train"):
    X_train = np.fromfile('./data/stl10_binary/' + prefix + '_X.bin', dtype=np.uint8)
    y_train = np.fromfile('./data/stl10_binary/' + prefix + '_y.bin', dtype=np.uint8)

    X_train = np.reshape(X_train, (-1, 3, 96, 96)) # CWH
    X_train = np.transpose(X_train, (0, 1, 3, 2)) # CHW

    return X_train, y_train - 1

def eval_trail(model, X_train, y_train, X_test, y_test, config):
    X_train_feature = []

    for batch_x, batch_y in next_batch(X_train, y_train, batch_size=config['batch_size']):
        features, _ = model(batch_x)
        X_train_feature.extend(features.cpu().detach().numpy())

    X_train_feature = np.array(X_train_feature)
    
    X_test_feature = []

    for batch_x, batch_y in next_batch(X_test, y_test, batch_size=config['batch_size']):
        features, _ = model(batch_x)
        X_test_feature.extend(features.cpu().detach().numpy())

    X_test_feature = np.array(X_test_feature)
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_feature)

    train_acc, test_acc = linear_model_eval(scaler.transform(X_train_feature), y_train, scaler.transform(X_test_feature), y_test)
    
    return train_acc, test_acc

def linear_model_eval(X_train, y_train, X_test, y_test):
    
    clf = LogisticRegression(random_state=0, max_iter=1200, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)
    return clf.score(X_train, y_train), clf.score(X_test, y_test)