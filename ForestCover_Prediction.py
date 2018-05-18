# -*- coding: utf-8 -*-
"""
@author: Wong Yit Jian
@updated: 12/11/2017
"""
# Supress warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

def model(trainX,trainY,testX):
    clf = ExtraTreesClassifier(n_estimators = 100, n_jobs = -1).fit(trainX, trainY)
    prediction = clf.predict(testX)
    return prediction

# Read file
forestDataset_train = pd.read_csv('train_processed.csv')
forestDataset_test = pd.read_csv('test_processed.csv')

# Partition training and test dataset
trainX = forestDataset_train.iloc[:,0:63]   # Training features
trainY = forestDataset_train.iloc[:, 63:]   # Training label
testX = forestDataset_test                  # Test features

# Get test Id
testDataframe = pd.read_csv('test.csv')
testId = testDataframe['Id'].values

print('Predicting. . .')
prediction = model(trainX, trainY, testX)
submission = pd.DataFrame({'Id':testId, 'Cover_Type':prediction}).reindex(columns=['Id','Cover_Type'])
submission.to_csv('submission.csv', index = False)
print(' Done!\n')