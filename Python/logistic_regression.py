#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:30:04 2022

@author: natdeacon
"""

#####################################
#import packages
#####################################
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#####################################
#prepare basetable ('df')
#####################################
df = pd.read_csv("lsdata.csv", skiprows=1, skipinitialspace=True)
df["sent_at"] = pd.to_datetime(df["sent_at"].fillna("1900-01-01"), format= '%Y-%m-%d')
df = df[df["sent_at"]>"2019-08-01"]
df = df[df["school_name"] == "Georgetown University"]
df = df[df["lsat"] > 120] #likely fake entries; score equivilent to filling out no questions
df = df[["sent_at", "is_in_state", "is_fee_waived", "lsat",
        "softs", "urm", "non_trad", "gpa", "is_international", "years_out",
        "is_military", "is_character_and_fitness_issues", "simple_status"]]
accepted_status = ["Accepted", "WL, Accepted", "Hold, Accepted, Withdrawn", "WL, Accepted, Withdrawn"]
df["sent_month"] = pd.DatetimeIndex(df['sent_at']).month

#define function to calculate months after september applicants sent application
def months_after_open(sent_month):
    if sent_month > 6:
        return(sent_month-9) 
    else:
        return(3 + sent_month) #3 for oct+nov+dec, plust months into year

df["softs"] = df["softs"].str.strip("T")
df["softs"] = pd.to_numeric(df["softs"])
df["months_after_sept_sent"] = df['sent_month'].map(months_after_open)
df["is_military"].fillna(False,inplace=True)
df["is_military"] = df['is_military'].astype(str).map({'True': True, 'False': False}).astype("bool")
df["is_character_and_fitness_issues"].fillna(False,inplace=True)
df["is_character_and_fitness_issues"] = df['is_character_and_fitness_issues'].astype(str).map({'True': True, 'False': False}).astype("bool")

df["was_accepted"] = np.where(df["simple_status"].isin(accepted_status), 1, 0) 

df = df.drop(["sent_at", "sent_month", "simple_status"], axis = 1)
df = df.dropna(axis = 0)

#####################################
#split between training and testing sets
#####################################

# Create dataframes with variables and target
X = df.drop(["was_accepted"], axis = 1)
y = df["was_accepted"]

#70/30 split, stratify by y so train and test sets have equal target incidence
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, stratify = y)


#create final train and test dataframes
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

#####################################
#forward stepwise variable selection to determine order variables will be added to model
#####################################

#order candidate variables by AUC based on our original dataframe

#function to calculate auc
def auc(variables, target, basetable):
    X = basetable[variables]
    y = basetable[target]
    logreg = linear_model.LogisticRegression()
    logreg.fit(X, y)
    predictions = logreg.predict_proba(X)[:,1]
    auc = roc_auc_score(y, predictions)
    return(auc)

#function to calculate next best variable to add in terms of impoving auc of model
def next_best(current_variables,candidate_variables, target, basetable):
    best_auc = -1
    best_variable = None
    for v in candidate_variables:
        auc_v = auc(current_variables + [v], target, basetable)
        if auc_v >= best_auc:
            best_auc = auc_v
            best_variable = v
    return best_variable


#stepwise procedure
candidate_variables = df.columns.tolist()
candidate_variables.remove("was_accepted")

current_variables = []

number_iterations = len(candidate_variables)
for i in range(0, number_iterations):
    next_variable = next_best(current_variables, candidate_variables, ["was_accepted"], train)
    current_variables = current_variables + [next_variable]
    candidate_variables.remove(next_variable)
print(current_variables)


#####################################
#remove variables that are highly correlated with other variables
#####################################
iteration = 0
Var1 = []
Var2 = []
cor = []

for x in current_variables:
    iteration += 1
    for i in current_variables[iteration:]:
        correlation = np.corrcoef(train[x], train[i])[0,1]
        Var1.append(x)
        Var2.append(i)
        cor.append(correlation)
corrTable = pd.DataFrame()
corrTable["var1"] = Var1
corrTable["var2"] = Var2
corrTable["correlation"] = cor

# remove non traditional due to multi-collinearity with years out

current_variables.remove("non_trad")

#####################################
#compare train/test auc curves to determine cutoff of variables
#####################################

# Keep track of train and test AUC values
auc_values_train = []
auc_values_test = []
variables_evaluate = []

def auc_train_test(variables, target, train, test):
    X_train = train[variables]
    X_test = test[variables]
    Y_train = train[target]
    Y_test = test[target]
    logreg = linear_model.LogisticRegression()
    
    # Fit the model on train data
    logreg.fit(X_train, Y_train)
    
    # Calculate the predictions both on train and test data
    predictions_train = logreg.predict_proba(X_train)[:,1]
    predictions_test = logreg.predict_proba(X_test)[:,1]
    
    # Calculate the AUC both on train and test data
    auc_train = roc_auc_score(Y_train, predictions_train)
    auc_test = roc_auc_score(Y_test,predictions_test)
    return(auc_train, auc_test)

#Reorder to see if that changes anything
current_variables = ['is_fee_waived','lsat', 'softs', 'is_character_and_fitness_issues', 'gpa','is_international', 'years_out', 'urm',
 'is_military', 'months_after_sept_sent', 'is_in_state']

# Iterate over the variables in current_variables
for v in current_variables:
    # Add the variable
    variables_evaluate.append(v)
    # Calculate the train and test AUC of this set of variables
    auc_train, auc_test = auc_train_test(variables_evaluate, ["was_accepted"], train, test)
    # Append the values to the lists
    auc_values_train.append(auc_train)
    auc_values_test.append(auc_test)
    
#visualize auc of train vs. test datasets
x = np.array(range(0,len(auc_values_train)))
y_train = np.array(auc_values_train)
y_test = np.array(auc_values_test)
plt.xticks(x, current_variables, rotation = 90)
plt.plot(x,y_train)
plt.plot(x,y_test)
plt.ylim((0.5, 0.9))
plt.legend(labels = ["y_train", "y_test"])
plt.ylabel("AUC")
plt.show()

#select all variables before AUC of test line peaks
predictors = ["lsat", "gpa", "urm"]


#####################################
#constructing model
#####################################

X = df[predictors] #select predictor variables

y = df[["was_accepted"]] #select target variable

#create logistic regression model 
logreg = linear_model.LogisticRegression() 
#fit model to the data
logreg.fit(X, y)

#####################################
#intetpreting model
#####################################

# priningt coeficients 
coef = logreg.coef_
for p,c in zip(predictors,coef[0]):
    print(p + '\t' + str(c))
#print intercept
print(logreg.intercept_)

#####################################
#using model to make predictions
#####################################

new_df = df[predictors] #create df with just predictors

#create predictions, a list of probability pairs f0r each row of the df
predictions = logreg.predict_proba(new_df) 
#print first five
print(predictions[0:5])

predDf = pd.DataFrame([[175,3.49, True]], columns=["lsat", 'gpa', 'urm'])
logreg.predict_proba(predDf) 
#####################################
#cumulative gains curve
#####################################

#####################################
#lift curve
#####################################

