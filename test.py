# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:18:03 2019

@author: dell
"""

import pandas as pd
import numpy as np
import pickle
import time
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\dell\\Downloads\\DataSets\\Test.csv")
date_time=df["date_time"].values
dateTimeSplits = df["date_time"].str.split(" ")
df["Year"] = pd.DataFrame([ item[0].split("-")[0] for item in dateTimeSplits ])
df["Month"] = pd.DataFrame([ item[0].split("-")[1] for item in dateTimeSplits ])
df["Day"] = pd.DataFrame([ item[0].split("-")[2] for item in dateTimeSplits ])
df["hour"] = pd.DataFrame([ item[1].split(":")[0] for item in dateTimeSplits ])

df.drop(columns =["date_time"], inplace = True)
column_tuples = [
    ('is_holiday', LabelEncoder()),
    (['air_pollution_index'], StandardScaler()),
    (['humidity'], StandardScaler()),
    (['wind_speed'], StandardScaler()),
    (['wind_direction'], StandardScaler()),
    (['visibility_in_miles'], StandardScaler()),
    (['dew_point'], StandardScaler()),
    (['temperature'], StandardScaler()),
    (['rain_p_h'], StandardScaler()), 
    (['snow_p_h'], StandardScaler()),
    (['clouds_all'], StandardScaler()),
    ('weather_type', LabelEncoder()),
    ('weather_description', LabelEncoder()),
    (['Year'], StandardScaler()),
    (['Month'], MinMaxScaler()),
    (['Day'], MinMaxScaler()),
    (['hour'], MinMaxScaler())
]
mapper = DataFrameMapper(column_tuples,df_out=True)

mapper_fit = mapper.fit(df)
X = mapper_fit.transform(df)
X = X.values

mlp = pickle.load(open('mlp.pkl', 'rb'))
pred=mlp.predict(X)
prediction = [int(round(item)) for item in pred]
data = {'date_time': date_time,'traffic_volume':prediction}
filename = "Prediction{}.csv".format(int(time.time()))
output = pd.DataFrame(data)
output.to_csv(filename, index=False)