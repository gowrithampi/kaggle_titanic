# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:55:18 2020

@author: gthampi
"""
import zipfile 
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import xgboost as xgb
#from sklearn.ensemble import RandomForestClassifier as rnd
from itertools import product as pdt
from h2o.estimators import H2ORandomForestEstimator as rnd
import h2o as h2o 

h2o.init()

sns.set(style="whitegrid")

##working directory I wish to be in
#print(os.getcwd())
#os.chdir(r'C:\Users\gthampi\Documents\Python ML\titanic')
#print(os.getcwd())

##authenticate your information - kaggle.json should be available in C:/User/.kaggle/
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
#api.authenticate()


#download the competition data
#api.competition_download_files('titanic', r'C:\Users\gthampi\Documents\Python ML\titanic')
#print(os.getcwd())
#zipref = zipfile.ZipFile('titanic.zip')
#zipref.extractall()
#print(os.listdir())

gender_submission = pd.read_table('gender_submission.csv')
test = pd.read_table('test.csv', sep = ',')
train = pd.read_table('train.csv', sep = ',')



### Exploratory data analysis
print(train.dtypes)
print(train['Survived'].value_counts())

total_survivors = sum(train['Survived'])
total_fatalities = sum(train['Survived']==0)
train['fold'] = np.random.randint(1, 6, train.shape[0])
##feature engineering 

def feature_eng(X):
    
   X['title'] = X['Name'].str.contains("Dr.|Earl|Count|Countess|Rev|Lady|Lord").astype(int)
   X['married_woman'] = X['Name'].str.contains("Mrs").astype(int)
   X['child'] = (X['Age'] < 15).astype(int)
   X['Ticket'] = X['Ticket'].str.lower()
   X['Ticket'] = X['Ticket'].str.replace('.', '')
   X['Ticket'] = X['Ticket'].str.replace('/', '')
   X['ticket_alpha'] = X['Ticket'].str.extract(r'([\w].*?) ')
   X['ticket_alpha'] = X['ticket_alpha'].apply(lambda Y: 'Not Available' if Y!=Y else Y)
   X['Cabins'] = X['Cabin'].str.split()
   ##Y! =Y checks if Y is nan
   X['NumberCabins'] = X['Cabins'].apply( lambda Y : 'Not Available' if Y!=Y else len(Y))
   X['lettercabin'] = X['Cabin'].apply(lambda Y: 'Not Available' if Y!=Y else Y[0])
   X.drop(['Name'  , 'Cabin' , 'Cabins', 'Ticket'], axis =1, inplace = True)
   
   ##for creating cross validation folds   
   
   return X

def impute_missing(X):
   age_impute = X.groupby(['Sex','Pclass']).agg({'Age':'median'})
   age_impute.reset_index(level = ['Sex' , 'Pclass'], inplace = True)
   X =X.merge(age_impute, how = 'left', left_on = ['Sex' , 'Pclass'] , right_on = ['Sex' , 'Pclass'])
   X['Age_x'] = X.apply(lambda x: x['Age_y'] if pd.isnull(x['Age_x']) else x['Age_x'], axis=1)
   X.drop('Age_y', axis = 1, inplace = True)
   X.rename(columns = {'Age_x' : 'Age'}, inplace = True)
   return X 

train_crossval = train.copy()
train = feature_eng(train)
train = impute_missing(train)



categoricals = ['Pclass', 'Sex', 'Embarked' ,'title', 'child']
numericals = ['Age', 'SibSp' , 'Parch' , 'Fare']
probables = ['PassengerId', 'Name' , 'Ticket' , 'Cabin']



##analyze bar plots and percentages
for cat in categoricals :
    df = train.groupby(cat).agg({'Survived' : ['sum' ,'count']})    
    df.columns = df.columns.droplevel(0)    
    df['percent'] = df['sum']/df['count']
    ax = sns.barplot(x= df.index , y= 'percent', data= df)
    plt.show()

##analyze contingency tables
for cat in categoricals : 
    df = train.groupby([cat,'Survived'])['Survived'].count()
    c = train[cat].nunique()
    chi2 = chi2_contingency(np.array(df.to_list()).reshape(2,c))
    print(df)
    print(chi2[0])
    
    

for num in numericals : 
    sns.boxplot( x = 'Survived' , y = num , data = train)
    plt.show()
    


### data cleaning, exploratory analysis and feature engineering done, we are good to go with crossvalidation 
max_depth = [50, 100, None]
#n_estimators = [50, 100, 150]
ntrees = [50, 100, 150]
#min_samples_leaf = [1, 2, 4]
min_row = [1, 2, 4]

params_grid = list(pdt(max_depth, ntrees, min_row))

clf = rnd(balance_classes=True)


def crossvalidate(X, params_grid):
    
    av_accuracy = []
    total_params = len(params_grid)
    ###let's do crossvalidation for each set of parameters
    for param in params_grid: 
        sum_accuracy = 0
        average_accuracy = 0
        titanic_forest = rnd(max_depth = param[0] , ntrees = param[1] , min_rows = param[2])  
    
    ## we'll train where fold isn't X and test where fold is k hooray! 
        for k in range(1,6): 
        
             Xcross = X.loc[X['fold'] != k ]
             Xval = X.loc[X['fold']==k]
             Xcross = impute_missing(Xcross)
             Xcross = feature_eng(Xcross)
             Xval = impute_missing(Xval)
             Xval = feature_eng(Xval)
             
    ## what we wish to predict
             response_column = 'Survived'
    #information we have 
             train_columns = list(Xcross.columns).remove('Survived', 'PassengerId')           
    
    #convert to h2o dataframe
    
             X_h2o = h2o.H2OFrame(Xcross)   
             Xval_h2o = h2o.H2OFrame(Xval)
             Yval = Xval['Survived']
             titanic_forest.train(x = train_columns , y = response_column, training_frame=X_h2o)
             perf = titanic_forest.model_performance(Xval_h2o)
             predict_titanic = titanic_forest.predict(Xval_h2o)
             predict_titanic = predict_titanic.as_data_frame()
             predict_titanic = predict_titanic['predict'].to_list()
             predict_titanic = list(map( lambda x: 0 if x< 0.5 else 1, predict_titanic))
             accuracy = 1 - sum(abs(predict_titanic - Yval))/len(Xval_h2o)
             sum_accuracy = sum_accuracy + accuracy
             h2o.remove(X_h2o)
             h2o.remove(Xval_h2o)
        average_accuracy = sum_accuracy/5
        print(average_accuracy, param)
        av_accuracy.append(average_accuracy)    
    return av_accuracy
# Define model



    
    
    
   
    
        
#### imputing variables will be done within the cross validation fold to prevent data leakage
    

