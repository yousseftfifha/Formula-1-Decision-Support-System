#!/usr/bin/env python
# coding: utf-8

# ## Qualifying Fact Analysis

# #### Import

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pip
import geopandas
import altair as alt
from descartes.patch import PolygonPatch
from datetime import datetime
from apyori import apriori 
import plotly.express as px
from sklearn.svm import SVC
import warnings
warnings.simplefilter("ignore")

# importing ML libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import pickle


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


# ##### Deleting data before 2014

# Due to the lack of data accuracy before 2014 and the constant changes occurring on the formula one rules we saw best to only consider data from 2014 and above.

# In[8]:


Fact=Fact.drop(Fact[(Fact["year"] <=2014)].index)


# ##### Cleaning

# In[9]:


Fact["nameDriver"]=Fact["ForeName"]+" "+Fact["SurName"]
Fact["nameDriver"]
Fact['Dob']=pd.to_datetime(Fact['Dob'])
date=datetime.today()-Fact['Dob']
Fact['age']=round(date.dt.days/365)


# Calculating the drivers' age is more significant that having a birth date as it provides us a mesurable value.

# In[10]:


Fact.drop(columns=['raceId',
                   'DriverID',
                   'constructorId',
                   'CircuitId',
                   'dateId',
                   'constructorRef',
                   'DriverRef',
                   'circuitRef',
                   'ForeName',
                   'SurName',
                   'Dob',
                   'Number',
                    'day',
                   'month'],axis=1,inplace=True)


# In[11]:


Fact.rename(columns = {'name_x':'nameGP','name_y':'nameCircuit'}, inplace = True)


# In[12]:


Fact.info()


# In[13]:


Fact.shape


# In[14]:


Fact.head()


# In[15]:


Fact.tail()


# In[16]:


Fact.columns


# In[17]:


Fact["rank"].unique()


# In[18]:


selected_columns = Fact[["Code","rank","fastest_lapspeed","age","Nationality","points_y","year"]]
df = selected_columns.copy()
df


# In[19]:


df['winning']=0
for i in range(0,len(df)):
    pos=df['rank'].iat[i]
    if pos>0 and pos<10:
         df['winning'].iat[i] = int(1)
    else:
        
         df['winning'].iat[i] = int(0)


# In[20]:


df.drop(columns=['rank'],axis=1,inplace=True)


# In[21]:


df1=df
df1=df1.groupby(["Code"]).sum('winning')


# In[22]:


df1["performance"]=df1["winning"]/Fact.groupby(["Code"]).size()


# In[23]:


df=df1["performance"].to_frame()


# In[24]:


df1 = pd.merge(df,driver, on='Code', how='inner')
df1['Dob']=pd.to_datetime(df1['Dob'])
date=datetime.today()-df1['Dob']
df1['age']=round(date.dt.days/365)
df1.drop(columns=['DriverID','DriverRef','Number','ForeName','SurName','Dob','car'],axis=1,inplace=True)
df1['points'] = df1['points'].fillna(0).astype(np.int64, errors='ignore')
df1['age'] = df1['age'].fillna(0).astype(np.int64, errors='ignore')
df=df1
df


# In[25]:


df.sort_values(by="performance",ascending=False).head(15)


# In[26]:


le = LabelEncoder()
df["Code"] = le.fit_transform(df["Code"])
df["Nationality"] = le.fit_transform(df["Nationality"])


# In[27]:


df.sort_values(by="performance",ascending=False).head(15)


# In[28]:


X = df.drop(['performance'], axis=1)
y = df[['performance']]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=3)


# In[29]:


regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)


# In[30]:


test={'Code':['11'],'Nationality':['24'],'points':['170'],'age':['30']}
df=pd.DataFrame(test)
pred = regressor.predict(df)
print(pred)


# In[31]:


pickle.dump(regressor, open('performancePredictorInRace.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('performancePredictorInRace.pkl','rb'))

