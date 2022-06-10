#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:14:31 2022

@author: natdeacon
"""

###################################################
#Step 1: loading packages and data
###################################################
import pandas as pd
import numpy as np

df = pd.read_csv("employment-data.csv")

###################################################
#Step 2: basic data exploration
###################################################

#view columns names
df.columns
#view columns names and data types
df.dtypes
#see counts of values within a variable
df["Period"].value_counts()
#see unique of values within a variable
df["Period"].unique()

###################################################
#Step 3: cleaning data
###################################################

#date conversion (see ymd function documentation)
#convert column from number to character to date 
df["Period"]=df["Period"].astype(str)
df["Period"] = pd.to_datetime(df["Period"])

#filter df on conditions
df = df[(df["Series_title_2"] == "Agriculture, Forestry and Fishing") |(df["Series_title_2"] == "Mining")]
#select columns
df = df[["Period", "Data_value", "Subject", "Group", "Series_title_1", "Series_title_2"]]
#drop column
df = df.drop("Series_title_1", 1)

#add calculated columns 
df["Data_value_sqrt"] = np.sqrt(df["Data_value"])

#remove rows with null values 
df.dropna(axis = 1)
#remove columns containing only  null values
df.dropna(axis = 0)

###################################################
#Step 4: aggregating data
###################################################

#reset index removes the multi-index
dfSummary = df.groupby('Series_title_2').agg({'Data_value': 'mean'}).reset_index().copy()
dfSummary= dfSummary.rename(columns={"Data_value": "Data_value_mean"})
