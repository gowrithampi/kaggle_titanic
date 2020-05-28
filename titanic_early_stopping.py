# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:57:52 2020

@author: gthampi
"""
import pandas as pd
import zipfile 
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier as rnd
from itertools import product as pdt
nfold = 10
sns.set(style="whitegrid")
train = pd.read_table('train.csv', sep = ',')
train['fold'] = np.random.randint(1, nfold+1, train.shape[0])
categorical_features = [ 'Embarked','Sex','title', 'Age_bin', 'married_woman']#,'ticket_alpha', 'NumberCabins' , 'lettercabin']
sns.set(style="white")

def feature_eng(X):
   #age_cuts= ['child', 'youth', 'adult', 'senior']
   age_cuts= [0, 1, 2, 3]
   age_bins = [0,15,25,60,100]
   X['Age_bin'] = pd.cut(X['Age'], bins=age_bins, labels=age_cuts) 
   X['Age_bin'] = X['Age_bin'].astype(int)
   X['title'] = X['Name'].str.contains("Dr.|Earl|Count|Countess|Rev|Lady|Lord|Sir|Don|Dona|Rev|Major|Capt|Col|Jonkheer").astype('category')
   X['married_woman'] = X['Name'].str.contains("Mrs").astype('int')
   
   X['Ticket'] = X['Ticket'].str.lower()
   X['Ticket'] = X['Ticket'].str.replace('.', '')
   X['Ticket'] = X['Ticket'].str.replace('/', '')
   X['ticket_alpha'] = X['Ticket'].str.extract(r'([\w].*?) ')
   X['ticket_alpha'] = X['ticket_alpha'].apply(lambda Y: 'Not Available' if Y!=Y else Y)
   X['Cabins'] = X['Cabin'].str.split()
   ##Y! =Y checks if Y is nan
   X['NumberCabins'] = X['Cabins'].apply( lambda Y : 0 if Y!=Y else 1)
   X['lettercabin'] = X['Cabin'].apply(lambda Y: 'Not Available' if Y!=Y else Y[0])
   X['Embarked'] = X['Embarked'].apply(lambda Y: 'C' if Y!=Y else Y)
   X['Embarked'] = X['Embarked'].astype('category')
   X['Sex'] = X['Sex'].apply(lambda Y : 1 if Y =='female' else 0)
   X['Sex'] = X['Sex'].astype('int')
   X['ticket_alpha'] = X['ticket_alpha'].astype('category')
   #X['NumberCabins'] = X['NumberCabins'].astype('category')
   X['lettercabin'] = X['lettercabin'].astype('category')
   X['Family'] = X['Parch'] + X['SibSp'] + 1
   
   X.drop([ 'Fare','Age','Parch', 'SibSp', 'Name'  , 'Cabin' , 'Cabins',
           'Ticket' ,'PassengerId', 'lettercabin', 'ticket_alpha'], axis =1, inplace = True)
   ##for creating cross validation folds   
   
   return X

def impute_missing(X):
   age_impute = X.groupby(['Sex' , 'Pclass']).agg({'Age':'median'})
   age_impute.reset_index(level = ['Sex', 'Pclass'] , inplace = True)
   X =X.merge(age_impute, how = 'left', left_on = ['Sex' , 'Pclass']  , right_on = ['Sex' , 'Pclass' ])
   X['Age_x'] = X.apply(lambda x: x['Age_y'] if pd.isnull(x['Age_x']) else x['Age_x'], axis=1)
   X.drop('Age_y', axis = 1, inplace = True)
   X.rename(columns = {'Age_x' : 'Age'}, inplace = True)
   return X 






             
def crossvalidate(X):
    
    av_crossval_accuracies = []
    av_training_accuracies = []
    Xtrain = X.copy()
    Ytrain = Xtrain['Survived']
    Xtrain = Xtrain.drop(['fold', 'Survived'], axis = 1)
    Xtrain = impute_missing(Xtrain)
    Xtrain = feature_eng(Xtrain)
    train_data = lgb.Dataset(Xtrain, label=Ytrain, categorical_feature=categorical_features)
    
    
    
   ###let's do crossvalidation for each num trees
    for numtrees in range(1,100): 
        sum_crossval_accuracy = 0
        average_crossval_accuracy = 0
        
        train_accuracy = 0
        
        params = { 'feature_fraction' : 0.9, 'learning_rate' : 0.5, 'num_leaves' : 20, 'max_depth' : 4 ,'num_trees' : numtrees, 'boosting_type' : 'gbdt', 'metric' : 'binary_error', 'objective' : 'binary'}
        model_train = lgb.train(params = params,train_set = train_data , categorical_feature=categorical_features)
        predict_train = model_train.predict(Xtrain, categorical_feature=categorical_features)
        predict_train = list(map(lambda x : 0 if x< 0.5 else 1, predict_train))
        error_train = abs(predict_train - Ytrain)
        train_accuracy = 1 - sum(error_train)/len(Ytrain)
       
        
        
       
## we'll train where fold isn't X and test where fold is k hooray! 
        for k in range(1,nfold+1): 
        
             Xcross = X.loc[X['fold'] != k ].copy()
             Ycross = Xcross['Survived']
             
             
             Xval = X.loc[X['fold']==k].copy()
             Yval = Xval['Survived']
             
             Xcross = Xcross.drop(['fold', 'Survived'], axis = 1)
             Xval = Xval.drop(['fold', 'Survived'], axis = 1)
             
             Xcross = impute_missing(Xcross)
             Xcross = feature_eng(Xcross)
             Xval = impute_missing(Xval)
             Xval = feature_eng(Xval)
             
             cross_data = lgb.Dataset(Xcross, label=Ycross, categorical_feature=categorical_features)
             valid_data = lgb.Dataset(Xval, label=Yval, categorical_feature=categorical_features)
             #Xval_data = lgb.Dataset(Xval,categorical_feature=categorical_features )
             params = { 'feature_fraction' : 0.9, 'learning_rate' : 0.5, 'num_leaves' : 20, 'max_depth' : 4 ,'num_trees' : numtrees, 'boosting_type' : 'gbdt', 'metric' : 'binary_error', 'objective' : 'binary'}
             
             model = lgb.train(params = params,train_set = cross_data , categorical_feature=categorical_features )
             
             predict_val = model.predict(Xval, categorical_feature=categorical_features, num_iteration = model.best_iteration)
             predict_val = list(map(lambda x : 0 if x< 0.5 else 1, predict_val))
             error_val = abs(predict_val - Yval)
             accuracy = 1 - sum(error_val)/len(Yval)
             sum_crossval_accuracy = sum_crossval_accuracy + accuracy           
        average_crossval_accuracy = sum_crossval_accuracy/(nfold)
        
        av_crossval_accuracies.append(average_crossval_accuracy)
        av_training_accuracies.append(train_accuracy)
        print('executing' , numtrees)
    return  av_training_accuracies, av_crossval_accuracies

av_train, av_crossval = crossvalidate(train) 

ax = sns.lineplot(x= list(range(1,100)), y= av_crossval)
sns.lineplot(x= list(range(1,100)), y= av_train, ax = ax)
ax.set(xlabel='number of trees', ylabel='accuracy')
plt.legend(title='Data', loc='lower right', labels=['Cross Validation', 'Training'])
plt.show()