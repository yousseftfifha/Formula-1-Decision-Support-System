#!/usr/bin/env python
# coding: utf-8

# ## RaceResult Fact Analysis

# #### Import

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pip
#import geopandas
import altair as alt
from descartes.patch import PolygonPatch
from datetime import datetime


# ### Data  Extraction

# #### Connection to database 

# In[2]:


def DbConnect():
    conn = psycopg2.connect(host="localhost",database="FormulaOne",port=5432,user='postgres',password='root')
    return conn


# In[3]:


circuit = pd.read_sql('select * from "CircuitDim"', con=DbConnect())
races= pd.read_sql('select * from "RaceDim"', con=DbConnect())
date= pd.read_sql('select * from "DateTimeDim"', con=DbConnect())
races= pd.read_sql('select * from "RaceDim"', con=DbConnect())
driver= pd.read_sql('select * from "DriverDim"', con=DbConnect())
constructor= pd.read_sql('select * from "ConstructorsDim"', con=DbConnect())
status= pd.read_sql('select * from "StatusDim"', con=DbConnect())
result= pd.read_sql('select * from "RaceResultFact"', con=DbConnect())


# ### Merging to a single DataFrame

# ##### Renaming Keys

# In[4]:


races.rename(columns = {'datetime_fk':'dateId'}, inplace = True)
races.rename(columns = {'circuit_fk':'CircuitId'}, inplace = True)
result.rename(columns = {'race_fk':'raceId'}, inplace = True)
result.rename(columns = {'driver_fk':'DriverID'}, inplace = True)
result.rename(columns = {'constructor_fk':'constructorId'}, inplace = True)
result.rename(columns = {'status_fk':'statusId'}, inplace = True)


# In[5]:


df1 = pd.merge(races,date, on='dateId', how='inner')
Racedim = pd.merge(df1,circuit, on='CircuitId', how='inner')
df2 = pd.merge(result,Racedim, on='raceId', how='inner')
df3 = pd.merge(df2,driver, on='DriverID', how='inner')
df4 = pd.merge(df3,constructor, on='constructorId', how='inner')
df5 = pd.merge(df4,status, on='statusId', how='inner')


# ### Data Anlysis

# ##### Preparing

# In[6]:


Fact=df5


# In[7]:


Fact.columns


# ##### Cleaning

# In[8]:


Fact["nameDriver"]=Fact["ForeName"]+" "+Fact["SurName"]
Fact["nameDriver"]
Fact['Dob']=pd.to_datetime(Fact['Dob'])
date=datetime.today()-Fact['Dob']
Fact['age']=round(date.dt.days/365)


# Calculating the drivers' age is more significant that having a birth date as it provides us a mesurable value.

# In[9]:


Fact.drop(columns=['raceId',
                   'DriverID',
                   'CircuitId',
                   'dateId',
                   'constructorRef',
                   'DriverRef',
                   'circuitRef',
                   'ForeName',
                   'SurName',
                   'Dob',
                   'Number',
                   'Code',
                    'day',
                   'month'],axis=1,inplace=True)


# In[10]:


Fact.rename(columns = {'name_x':'nameGP','name_y':'nameCircuit'}, inplace = True)


# In[11]:


Fact.info()


# In[12]:


Fact.shape


# In[13]:


Fact.head()


# In[14]:


Fact.tail()


# #####  Checking for null Data

# In[15]:


Fact.isnull().sum()*100/len(Fact)


# ##### Analysis

# Status 
# 

# ## Car performance: merge weather

# In[16]:


config=pd.read_csv("DE2-PI.csv",header=0,sep =',')
Factcar=pd.merge(Fact,config, on=["constructorId"], how='inner')
Factcar


# In[17]:


Factcar.rename(columns = {'constructorName_y':'constructorName'}, inplace = True)
Factcar.info()


# In[18]:


CarRS=Factcar.drop(columns=['nameGP',
                   'constructorNationality',
                   'img',
                    'statusId',
                   'points_x',
                   'points_x',
                    'Nationality',
                    'longitude',
                    'latitude',
                    'location',
                        'time',
                           'Altitude',
                           'date',
                            'country',
                            'age',
                            'penalty_fk',
                            'weather_fk',
                            'nameDriver',
                            'winingRate',
                            'displacement',
                            'laps',
                            'fastest_lapspeed',
                            'wins',
                            'laptime',
                            'pitstop',
                            'status',
                            'mpg',
                            'weight',
                            'round',
                            'car'
                           ])
#CarRS['pitstop']=CarRS['pitstop'].astype(str)
#CarRS['pitstop']=CarRS['pitstop'].replace(":", "", regex=True)
#CarRS['nameDriver']=CarRS['nameDriver'].replace(" ", "", regex=True)
#CarRS['pitstop']=CarRS['pitstop'].replace("None", "0", regex=True)


# In[19]:


CarRS['points_y'] = CarRS['points_y'].fillna(0).astype(np.int64, errors='ignore')

CarRS.isnull().sum()*100/len(CarRS)


# In[20]:


CarRS['winning']=0
for i in range(0,len(CarRS)):
    pos=CarRS['rank'].iat[i]
    if pos>0 and pos<10:
         CarRS['winning'].iat[i] = int(1)
    else:
        
         CarRS['winning'].iat[i] = int(0)


# In[21]:


CarRS.drop(columns=['rank'],axis=1,inplace=True)


# In[22]:


df1=CarRS
df1=CarRS.groupby(["constructorId"]).sum('winning')


# In[23]:


df1["performance"]=df1["winning"]/CarRS.groupby(["constructorId"]).size()


# In[24]:


df=df1["performance"].to_frame()
df


# In[25]:


constructor


# In[26]:


df1 = pd.merge(df,constructor, on='constructorId', how='inner')
df2 = pd.merge(df1,config, on='constructorId', how='inner')


# In[27]:


CarRS=df2
CarRS=CarRS.drop(columns=['constructorId','constructorRef','constructorName_y','mpg','displacement','winingRate'
                           ])
CarRS.rename(columns = {'constructorName_x':'constructorName'}, inplace = True)
CarRS


# In[28]:


from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
CarRS["constructorName"] = lb_make.fit_transform(CarRS["constructorName"])
CarRS["constructorNationality"] = lb_make.fit_transform(CarRS["constructorNationality"])
CarRS


# # Regression Linéaire multiple : Car Performance

# ## $$f(X_1,X_2,...)= a_0 + a_1*X_1 + a_2*X_2 + .... + a_{12}*X12$$

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = CarRS.drop(['performance'], axis=1)
y = CarRS[['performance']]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)


# In[39]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
#Régression linéaire multiple (toutes les variables de Boston) 
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)
y_pred = regressor.predict(X_test)

print('test_score = ',regressor.score(X_test,y_test)) 
print('R2 = ',r2_score(y_test, y_pred))
print('MAE = ',mean_squared_error(y_test, y_pred))
print('RMSE = ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('MAE = ', mean_absolute_error(y_test, y_pred))
print('MeadianAE = ', median_absolute_error(y_test, y_pred))

print('Intercept = ', regressor.intercept_)
print('Coefficients : ',regressor.coef_)


# In[41]:


import pickle

pickle.dump(regressor, open('ConstructorperformancePredictorInRace.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('ConstructorperformancePredictorInRace.pkl','rb'))


# In[ ]:




