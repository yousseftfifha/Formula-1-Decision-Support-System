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
    conn = psycopg2.connect(host="localhost",database="FormulaOne",port=5432,user='postgres',password='Foufou2010.')
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


weather=pd.read_csv("C:/Users/DELL/Downloads/meteooo.csv",header=0,sep =',',encoding = 'latin1')
weather=weather.drop(columns=['details'])
weather


# In[17]:


weather.rename(columns = {'date':'year'}, inplace = True)


# In[18]:


Factcar1= pd.merge(Fact,weather, on=["nameGP","year"], how='inner')
config=pd.read_csv("C:/Users/DELL/Desktop/DE2-PI.csv",header=0,sep =',')
Factcar=pd.merge(Factcar1,config, on=["constructorId"], how='inner')
Factcar


# In[19]:


Factcar.rename(columns = {'constructorName_y':'constructorName'}, inplace = True)
Factcar.info()


# In[20]:


CarRS=Factcar.drop(columns=['nameGP',
                   'constructorNationality',
                   'img',
                    'statusId',
                        'constructorId',
                   'points_x',
                   'points_x',
                    'Nationality',
                    'longitude',
                    'latitude',
                    'location',
                        'time',
                           'Altitude',
                           'date',
                            'nameCircuit',
                            'country',
                            'age',
                            'penalty_fk',
                            'weather_fk',
                            'nameDriver',
                            'winingRate',
                            'displacement',
                            'rank',
                            'laps',
                            'fastest_lapspeed',
                            'wins',
                            'laptime',
                            'pitstop',
                            'round',
                            'year',
                            'status'
                           ])
CarRS['car']=CarRS['car'].replace("None", "0", regex=True)
#CarRS['pitstop']=CarRS['pitstop'].astype(str)
#CarRS['pitstop']=CarRS['pitstop'].replace(":", "", regex=True)
#CarRS['nameDriver']=CarRS['nameDriver'].replace(" ", "", regex=True)
#CarRS['pitstop']=CarRS['pitstop'].replace("None", "0", regex=True)


# In[21]:


CarRS.isnull().sum()*100/len(CarRS)


# In[22]:


CarRS.tail()


# In[23]:


from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
#CarRS["nameCircuit"] = lb_make.fit_transform(CarRS["nameCircuit"])
#CarRS["country"] = lb_make.fit_transform(CarRS["country"])
CarRS["weather"] = lb_make.fit_transform(CarRS["weather"])
CarRS["constructorName"] = lb_make.fit_transform(CarRS["constructorName"])
CarRS["car"] = lb_make.fit_transform(CarRS["car"])
#CarRS["status"] = lb_make.fit_transform(CarRS["status"])
#CarRS["nameDriver"] = lb_make.fit_transform(CarRS["nameDriver"])
#CarRS["laptime"] = lb_make.fit_transform(CarRS["laptime"])
#CarRS["fastest_lapspeed"] = lb_make.fit_transform(CarRS["fastest_lapspeed"])
#CarRS["pitstop"] = lb_make.fit_transform(CarRS["pitstop"])
#CarRS["nameDriver"] = lb_make.fit_transform(CarRS["nameDriver"])
CarRS.drop(columns=['constructorName_x'],axis=1,inplace=True)
CarRS


# In[24]:


CarRS.info()


# # Regression Linéaire multiple : Car Performance

# ## $$f(X_1,X_2,...)= a_0 + a_1*X_1 + a_2*X_2 + .... + a_{12}*X12$$

# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = CarRS.drop(['car','points_y'], axis=1)
y = CarRS[['points_y']]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)


# In[26]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
#Régression linéaire multiple (toutes les variables de Boston) 
linreg2 = LinearRegression(fit_intercept= True, normalize=False) 
#pour imposer une ordonnée à l'origine (Par défaut ='True')
# Normalisation des données normalize=True
linreg2.fit(X_train, y_train)
y_pred = linreg2.predict(X_test)

print('test_score = ',linreg2.score(X_test,y_test)) 
print('R2 = ',r2_score(y_test, y_pred))
print('MAE = ',mean_squared_error(y_test, y_pred))
print('RMSE = ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('MAE = ', mean_absolute_error(y_test, y_pred))
print('MeadianAE = ', median_absolute_error(y_test, y_pred))

print('Intercept = ', linreg2.intercept_)
print('Coefficients : ',linreg2.coef_)


# In[27]:


import statsmodels.api as sm
est1 = sm.OLS(y_test, X_test)
est2 = est1.fit()
print(est2.summary())


# In[28]:


dfDuplicated = CarRS[CarRS.duplicated()]

#print(CarRS)
CarRS[CarRS['constructorName']==10]


# In[29]:


df1=CarRS.drop_duplicates(subset=['constructorName'], keep='last')


# In[30]:


df1[df1['constructorName']==10]


# In[31]:


# Define the X (inputs) and y (target) features
X = df1.drop(['car','points_y'], axis=1)
y = df1['points_y']


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)


# In[33]:


import pickle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)
pickle.dump(regressor, open('modelz.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('modelz.pkl','rb'))
print(model.predict([[11, 5, 23,10,950,247]]))

