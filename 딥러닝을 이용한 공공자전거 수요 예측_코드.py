#!/usr/bin/env python
# coding: utf-8

# ### Load Dataset

# In[1]:


import numpy as np
import pandas as pd

bike=pd.read_csv("bike_data_최종5.csv",index_col="date",parse_dates=["date"],infer_datetime_format=True)

bike.head()


# In[2]:


bike=bike.rename(columns={"Temperature(℃)":"temp", "Humidity(%)":"Humidity","Wind speed(km/h)":"Windspeed","Micro dust(PM-2.5 (㎍/m3)":"microdust", "Number of rentals per day":"No"})

bike.head()


# ### Data exploration

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=(20,10)
plt.rcParams["lines.linewidth"]=2
plt.rcParams["lines.color"]="r"
plt.rcParams["axes.grid"]=True

plot_cols=["temp","Humidity","Windspeed","microdust"]
plot_features=bike[plot_cols]
_=plot_features.plot(subplots=True)


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(figsize=(8, 6))
plt.subplot(221)
sns.distplot(bike["No"])
plt.subplot(222)
sns.distplot(bike["microdust"])
plt.subplot(223)
sns.distplot(bike["Windspeed"])
plt.subplot(224)
sns.distplot(bike["temp"])


# In[39]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.scatter(bike['No'], bike["temp"])
ax1.set_xlabel("No")
ax1.set_ylabel("temp")

ax2.scatter(bike['No'], bike["microdust"])
ax2.set_xlabel("No")
ax2.set_ylabel("microdust")

ax3.scatter(bike['No'], bike["Windspeed"])
ax3.set_xlabel("No")
ax3.set_ylabel("Windspeed")

ax4.scatter(bike['No'], bike["Humidity"])
ax4.set_xlabel("No")
ax4.set_ylabel("Humidity")
plt.show()


# In[37]:


# count x month
plt.figure(figsize=(6, 4))
sns.lineplot(x='Month', y='No',data=bike)

plt.figure(figsize=(6, 4))
sns.lineplot(x='Year', y='No',data=bike)


# In[35]:


plt.figure(figsize=(7, 3))
sns.pointplot(x='Day of the week', y='No',hue='Day of the week',data=bike)
plt.legend(loc='upper left')

plt.figure(figsize=(7, 3))
sns.pointplot(x='Weekend vs weekday', y='No',hue='Weekend vs weekday',data=bike)

plt.figure(figsize=(7, 3))
sns.pointplot(x='Holiday vs working day', y='No',hue='Holiday vs working day', data=bike)


# ### Data preprocessing and feature engineering

# In[3]:


train_size=int(len(bike)*0.9)
test_size=len(bike)-train_size
train,test=bike.iloc[0:train_size],bike.iloc[train_size:len(bike)]

print(len(train),len(test))


# In[4]:


#some other scalers to note include MinMaxScaler, StandardScaler, RobustScaler, Nomalizer
#Since the RMSE when including weekend vs weekdays, day of the week, holiday vs working day is substantially big,
#the model was produced using the variables temperature, humidity, windspeed and microdust


from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

f_columns=["temp","Humidity","Windspeed","microdust"]

f_transformer=RobustScaler()
f_transformer=f_transformer.fit(train[f_columns].to_numpy())

train.loc[:,f_columns]=f_transformer.transform(train[f_columns].to_numpy())
test.loc[:, f_columns]=f_transformer.transform(test[f_columns].to_numpy())


# In[5]:


no_transformer=RobustScaler()

no_transformer=no_transformer.fit(train[["No"]])
train["No"]=no_transformer.transform(train[["No"]])
test["No"]=no_transformer.transform(test[["No"]])


# In[8]:


def create_dataset(X,y, time_steps=1):
    Xs, ys=[],[]
    for i in range(len(X)-time_steps):
        v=X.iloc[i:(i+time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
        
    return np.array(Xs),np.array(ys)


# In[9]:


time_steps=1

X_train, y_train=create_dataset(train, train.No, time_steps)
X_test, y_test=create_dataset(test, test.No, time_steps)

print(X_train.shape,y_train.shape)


# ### Modelling and experiment

# 1. Vanilla LSTM

# In[10]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[11]:


#dropout rate of 0.4, optimizer=adam, activation function="relu"
from tensorflow.keras.layers import LSTM

model=keras.Sequential()
model.add(LSTM(units=128,activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.Dense(units=1))
model.compile(loss="mean_squared_error", optimizer="adam")

history=model.fit(X_train, y_train, epochs=30, batch_size=50, validation_split=0.1, shuffle=False)


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=(20,10)
plt.plot(history.history["loss"],label="train")
plt.plot(history.history["val_loss"],label="test")
plt.legend();


# In[13]:


y_pred=model.predict(X_test)
y_train_inv=no_transformer.inverse_transform(y_train.reshape(1,-1))
y_test_inv=no_transformer.inverse_transform(y_test.reshape(1,-1))
y_pred_inv=no_transformer.inverse_transform(y_pred)


# In[14]:


plt.plot(np.arange(0,len(y_train)), y_train_inv.flatten(),"g",label="history")
plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)),y_test_inv.flatten(),marker=".",label="true")
plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)),y_pred_inv.flatten(),"r",label="prediction")
plt.ylabel("Bike Count")
plt.xlabel("Time step")
plt.legend()
plt.show()


# In[15]:


plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Bike Count')
plt.xlabel('Time Step')
plt.legend()
plt.show();


# In[16]:


#rmse

from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse
from math import *

err= rmse(y_test_inv.flatten(), y_pred_inv.flatten())
print('RMSE',sqrt(err))


# 2. Stacked LSTM

# In[17]:


from tensorflow.keras.layers import LSTM

model=keras.Sequential()
model.add(LSTM(units=128,activation="relu", return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=128, activation="relu"))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.Dense(units=1))
model.compile(loss="mean_squared_error", optimizer="adam")

history=model.fit(X_train, y_train, epochs=30, batch_size=50, validation_split=0.1, shuffle=False)

plt.plot(history.history["loss"],label="train")
plt.plot(history.history["val_loss"],label="test")
plt.legend();


# In[18]:


y_pred=model.predict(X_test)
y_train_inv=no_transformer.inverse_transform(y_train.reshape(1,-1))
y_test_inv=no_transformer.inverse_transform(y_test.reshape(1,-1))
y_pred_inv=no_transformer.inverse_transform(y_pred)


# In[19]:


plt.plot(np.arange(0,len(y_train)), y_train_inv.flatten(),"g",label="history")
plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)),y_test_inv.flatten(),marker=".",label="true")
plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)),y_pred_inv.flatten(),"r",label="prediction")
plt.ylabel("Bike Count")
plt.xlabel("Time step")
plt.legend()
plt.show()


# In[20]:


plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Bike Count')
plt.xlabel('Time Step')
plt.legend()
plt.show();


# In[21]:


err= rmse(y_test_inv.flatten(), y_pred_inv.flatten()).sum()
print('RMSE',sqrt(err))


# 3. Bidirectional LSTM

# In[22]:


model=keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,input_shape=(X_train.shape[1], X_train.shape[2]))))

model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.Dense(units=1))
model.compile(loss="mean_squared_error", optimizer="adam")


# In[23]:


history=model.fit(X_train, y_train, epochs=30, batch_size=50, validation_split=0.1, shuffle=False)


# In[24]:


plt.plot(history.history["loss"],label="train")
plt.plot(history.history["val_loss"],label="test")
plt.legend();


# In[25]:


y_pred=model.predict(X_test)
y_train_inv=no_transformer.inverse_transform(y_train.reshape(1,-1))
y_test_inv=no_transformer.inverse_transform(y_test.reshape(1,-1))
y_pred_inv=no_transformer.inverse_transform(y_pred)


# In[26]:


plt.plot(np.arange(0,len(y_train)), y_train_inv.flatten(),"g",label="history")
plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)),y_test_inv.flatten(),marker=".",label="true")
plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)),y_pred_inv.flatten(),"r",label="prediction")
plt.ylabel("Bike Count")
plt.xlabel("Time step")
plt.legend()
plt.show()


# In[27]:


plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Bike Count')
plt.xlabel('Time Step')
plt.legend()
plt.show();


# In[28]:


err= rmse(y_test_inv.flatten(), y_pred_inv.flatten()).sum()
print('RMSE',sqrt(err))


# In[ ]:




