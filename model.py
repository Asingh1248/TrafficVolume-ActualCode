import pandas as pd
import numpy as np
import pickle
import time
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\dell\\Downloads\\DataSets\\Train.csv")
Y=df['traffic_volume'].values
dateTimeSplits = df["date_time"].str.split(" ")
df["Year"] = pd.DataFrame([ item[0].split("/")[0] for item in dateTimeSplits ])
df["Month"] = pd.DataFrame([ item[0].split("/")[1] for item in dateTimeSplits ])
df["Day"] = pd.DataFrame([ item[0].split("/")[2] for item in dateTimeSplits ])
df["hour"] = pd.DataFrame([ item[1].split(":")[0] for item in dateTimeSplits ])
data = {'traffic_volume':df['traffic_volume'], 'date_time':df['date_time']}
target = pd.DataFrame(data).set_index('date_time')
'''plt.figure(figsize=(15,5))
plt.plot(target[:200])'''

df.drop(columns =["date_time","traffic_volume"], inplace = True)
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

#(train_feat,test_feat,train_classes,test_classes)=train_test_split(X,Y,train_size=0.7,random_state=1)

#reg=LinearRegression()
#reg.fit(train_feat,train_classes)
#pred=reg.predict(test_feat)

mlp = MLPClassifier(hidden_layer_sizes=(64,256,200),max_iter=100,activation = 'relu')
mlp.fit(X,Y)
pred = mlp.predict(X)
filename = "mlp.pkl{}".format(int(time.time()))
pickle.dump(mlp, open(filename, 'wb'))
model = pickle.load(open(filename,'rb'))

print(model.predict())
# pickle.load() method loads the method and saves the deserialized bytes to model. Predictions can be done using model.predict().


'''print("Accuracy: ",accuracy_score(Y,pred))
#print(test_classes," ",pred)
plt.figure(figsize=(15,5))
plt.plot(Y,label="Original Data")
plt.plot(pred,label="Predictions")
## plt.plot(rcpredList,label="RC Predictions")
plt.legend(loc='upper left')
plt.title("Comaprison of flow Between Original and predictions")

plt.figure(figsize=(15,5))
plt.plot(target[:200],label="Original Data")
plt.plot(pred[:200],label="Predictions")
# plt.plot(rcpredList,label="RC Predictions")
plt.legend(loc='upper left')
plt.title("Comaprison of flow Between Original and predictions")'''




