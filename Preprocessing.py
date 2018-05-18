# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:48:40 2017

@author: hello
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


####Functions for transforming aspect####
def aspect_transform(x):
    if x+180>360:
        return x-180
    else:
        return x+180
    
    
####Functions for plotting distribution of conver types on pari of attributes####
def plotc(c1,c2):

    fig = plt.figure(figsize=(16,8))
    sel = np.array(list(train.Cover_Type.values))

    plt.scatter(c1, c2, c=sel, s=100, cmap = plt.cm.jet, alpha = 0.7)
    plt.xlabel(c1.name)
    plt.ylabel(c2.name)


####read train and test data####
train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')

print(train.head(10))

####numerical values of data ####
####Uncomment the following part to view description####


#pd.set_option('display.max_columns', None)
#print(train.describe())


####Uncomment the following part to view barplots####


#cols = train.columns
#
#size = len(cols)-1
#x = cols[size]
#y = cols[0:size]
#for i in range(0,size):
#    sns.barplot(data=train,x=x,y=y[i])  
#    plt.savefig('bar_plot_'+str(i))
#    plt.show()


####################################################

#drop soil_Type7 and soil_Type15
train.drop('Soil_Type7', axis = 1, inplace = True)
train.drop('Soil_Type15', axis = 1, inplace = True)
train.drop('Id', axis = 1, inplace = True)



test.drop('Soil_Type7', axis = 1, inplace = True)
test.drop('Soil_Type15', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)


###Selct 10 pairs of highest correlation
corr = train.corr().abs()
#heatmap for all paris
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

unstack = corr.unstack()
sort = unstack.sort_values(kind = "quicksort")

topTen = sort[-74:-54]
ttUnstack = topTen.unstack()
for idx, row in ttUnstack.iterrows():
    row[idx] = 1.0

#heatmap for 10 pairs of highest correlation

sns.heatmap(ttUnstack, xticklabels=ttUnstack.columns.values, yticklabels=ttUnstack.columns.values)


####Creating new features from numerical description of traning dataset####

train['Aspect'] = train.Aspect.map(aspect_transform)
test['Aspect'] = test.Aspect.map(aspect_transform)

train['HigherThan3100'] = train.Elevation > 3100 *3
test['HigherThan3100'] = test.Elevation > 3100 *3

train.HigherThan3100 = train.HigherThan3100.astype(int)
test.HigherThan3100 = test.HigherThan3100.astype(int)

train['Highwater'] = train.Vertical_Distance_To_Hydrology < 0
test['Highwater'] = test.Vertical_Distance_To_Hydrology < 0

train.Highwater = train.Highwater.astype(int)
test.Highwater = test.Highwater.astype(int)


####Plotting the distribution of covertypes on every pair-combinations of features####


# for i in range(10):
#     for j in range(i,10):
#         plotc(train.iloc[:,i], train.iloc[:,j])
    
    
####################################################



####Feature Engineering####

train['EleVerDis2Hydro'] = train.Elevation-train.Vertical_Distance_To_Hydrology
test['EleVerDis2Hydro'] = test.Elevation-test.Vertical_Distance_To_Hydrology

train['EleHorDis2Hydro'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2
test['EleHorDis2Hydro'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2

train['Distanse_to_Hydrolody'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
test['Distanse_to_Hydrolody'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5

train['Hydro_Fire_1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
test['Hydro_Fire_1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']

train['Hydro_Fire_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
test['Hydro_Fire_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

train['Hydro_Road_1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])

train['Hydro_Road_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

train['Horizontal_Distance_To_Hydrology'] =np.sqrt(train['Horizontal_Distance_To_Hydrology'])
test['Horizontal_Distance_To_Hydrology'] =np.sqrt(test['Horizontal_Distance_To_Hydrology'])

train['Horizontal_Distance_To_Roadways'] =np.sqrt(train['Horizontal_Distance_To_Roadways'])
test['Horizontal_Distance_To_Roadways'] =np.sqrt(test['Horizontal_Distance_To_Roadways'])


####Correlation Heatmap after Feature Engineering####


train_corr = train[["Aspect","Elevation","Slope","EleVerDis2Hydro","Distanse_to_Hydrolody",\
                    "Hydro_Fire_1","Fire_Road_2","EleHorDis2Hydro","Hydro_Fire_2",\
                    "Hydro_Road_2","Fire_Road_2", "Hydro_Road_1","Hillshade_9am",\
                    "Hillshade_3pm","Hillshade_Noon"]]
corr = train_corr.corr().abs()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, square  =True)


#####Preparing Dataset for Normalization and Scalling#####
feature_cols = [col for col in train.columns if col not in ['Cover_Type']]


trainX = train[feature_cols].values
testX = test[feature_cols].values
train_Y = list(train.Cover_Type.values)

####The start of Categorical Data####
catStart = 19

trainX = np.roll(trainX, 9, axis = 1)
testX = np.roll(testX, 9, axis = 1)

feature_cols = np.roll(feature_cols, 9)
testX = np.nan_to_num(testX)



####Scaling and Normalization####
####MinMax Scalaer is selected as the optimal scaling method for this dataset####
####Therefore, the parts of Standard Scaler and Normalizer are commented out####

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
                
#scale non-catergorical data in train and test 

##MinMax
##Apply transform only for non-categorical data
tempX = MinMaxScaler().fit_transform(trainX[:,0:catStart])
temp_test = MinMaxScaler().fit_transform(testX[:,0:catStart])

#Concatenate non-categorical data and categorical
conX = np.concatenate((tempX,trainX[:,catStart:]),axis=1)
con_test = np.concatenate((temp_test,testX[:,catStart:]),axis=1)


trainX = conX
testX = con_test

train_Y = train['Cover_Type']
train = pd.DataFrame(data = trainX, columns = feature_cols)
train = train.assign(Cover_Type = pd.Series(train_Y).values)

test = pd.DataFrame(data = testX, columns = feature_cols)
#
print(train.head(10))
print(test.head(10))

#Adding more weights to three key features during prediction
train['Aspect'] = train['Aspect'] * 2
test['Aspect'] = test['Aspect'] * 2

train['Elevation'] = train['Elevation'] * 2.5
test['Elevation'] = test['Elevation'] * 2.5

train['Slope'] = train['Slope'] * 1.5
test['Slope'] = test['Slope'] * 1.5

#Output to csv file
train.to_csv('train_processed.csv', index = False, sep=',')
test.to_csv('test_processed.csv', index = False, sep=',')


###Standardized
###Apply transform only for non-categorical data
#tempX = StandardScaler().fit_transform(trainX[:,:catStart])
#temp_test = StandardScaler().fit_transform(testX[:,:catStart])
#
##Concatenate non-categorical data and categorical
#conX = np.concatenate((tempX,trainX[:,catStart:]),axis=1)
#con_test = np.concatenate((temp_test,testX[:,catStart:]),axis=1)
#
#trainX = conX
#testX = con_test



##Normalize
##Apply transform only for non-categorical data
#tempX = Normalizer().fit_transform(trainX[:,0:catStart])
#temp_test = Normalizer().fit_transform(testX[:,:catStart])
##Concatenate non-categorical data and categorical
#conX = np.concatenate((tempX,trainX[:,catStart:]),axis=1)
#con_test = np.concatenate((temp_test,testX[:,catStart:]),axis=1)
#
#trainX = conX
#testX = con_test