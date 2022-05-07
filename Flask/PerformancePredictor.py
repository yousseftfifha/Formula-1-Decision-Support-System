#!/usr/bin/env python
# coding: utf-8

# # <center>${\textbf{Top Potential Driver in Qualifying Session}}$</center>

# ${\textbf{Overview}}$

# A Formula One season consists of a series of races, known as Grands Prix, which take place worldwide on purpose-built circuits and on public roads. <br>
# A single competition takes place over three days, day one is a practice session ,the second day is the qualification round in which racers compete against time in order to obtain an advantageous position during  the third day which is the final race where drivers race against each other .<br>
# Most importantly for our case a  qualifying session is held before each race to determine the order cars will be lined up in at the start of the race, with the fastest qualifier starting at the front and the slowest at the back.<br>
# Currently, the first qualifying period (Q1) is eighteen minutes long, with all twenty cars competing.<br>
# At the end of Q1, the five slowest drivers are eliminated from further qualification rounds, and fill positions sixteen to twenty on the grid based on their fastest lap time. Any driver attempting to set a qualifying time when the period ends is permitted to finish his lap, though no new laps may be started once the chequered flag is shown. <br>
# After a short break, the second period (Q2) begins, with fifteen cars on the circuit.
# <br> At the end of Q2, the five slowest drivers are once again eliminated, filling grid positions eleven to fifteen. Finally, the third qualifying period (Q3) features the ten fastest drivers from the second period. 
# <br>The drivers are issued a new set of soft tyres and have twelve minutes to set a qualifying time, which will determine the top ten positions on the grid. The driver who sets the fastest qualifying time is said to be on pole position, the grid position that offers the best physical position from which to start the race.<br>

# ${\textbf{Importance of Data}}$

# The insight analysis drills down to all information on the Formula 1 races, drivers, constructors, qualifying, circuits, lap times, pit stops, championships from 1950 till the latest 2021 season.<br>
# With the amount of data being captured, analyzed and used to design, build and drive the Formula 1 cars is astounding. It is a global sport being followed by millions of people worldwide and it is very fascinating to see drivers pushing their limit in these vehicles to become the fastest racers in the world!<br>

# ${\textbf{Import Libraries}}$

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


# ## ${\textbf{Part 1 - First Steps  }}$

# >${\textbf{Connect with Database}}$

# Previously on the integration Phase we have used Talend Open Studio for Intergration Services linkes with a PostgresSQL .<br>So now it's only logical that we connect python with the postgres Database in order to retrieve our dimensional data

# In[2]:


def DbConnect():
    conn = psycopg2.connect(host="localhost",database="FormulaOne",port=5432,user='postgres',password='root')
    return conn


# >${\textbf{Get data from QualifyingFact and related tables}}$

# In[3]:


weather = pd.read_sql('select * from "WeatherDim"', con=DbConnect())
date= pd.read_sql('select * from "DateTimeDim"', con=DbConnect())
circuit = pd.read_sql('select * from "CircuitDim"', con=DbConnect())
races= pd.read_sql('select * from "RaceDim"', con=DbConnect())
driver= pd.read_sql('select * from "DriverDim"', con=DbConnect())
constructor= pd.read_sql('select * from "ConstructorsDim"', con=DbConnect())
qualifyingFact= pd.read_sql('select * from "QualifyingFact"', con=DbConnect())


# >${\textbf{Merging Data to a single DataFrame }}$

# Renaming certain columns is essential for the merging 

# In[4]:


races.rename(columns = {'datetime_fk':'dateId'}, inplace = True)
races.rename(columns = {'weather_fk':'weatherID'}, inplace = True)
races.rename(columns = {'circuit_fk':'CircuitId'}, inplace = True)
qualifyingFact.rename(columns = {'race_fk':'raceId'}, inplace = True)
qualifyingFact.rename(columns = {'driver_fk':'DriverID'}, inplace = True)
qualifyingFact.rename(columns = {'constructor_fk':'constructorId'}, inplace = True)


# In[5]:


df1 = pd.merge(races,date, on='dateId', how='inner')
df2 = pd.merge(df1,circuit, on='CircuitId', how='inner')
df3 = pd.merge(df2,weather, on='weatherID', how='inner')
df4 = pd.merge(qualifyingFact,df3, on='raceId', how='inner')
df5 = pd.merge(df4,driver, on='DriverID', how='inner')
Fact = pd.merge(df5,constructor, on='constructorId', how='inner')


# ## ${\textbf{Part 2 - Exploratory Data Analysis  }}$

# > ${\textbf{Renaming columns }}$

# In[6]:


Fact.rename(columns = {'name_x':'nameGP','name_y':'nameCircuit','Nationality':'driverNationality'}, inplace = True)


# > ${\textbf{Converting Date of birth to AGE}}$

# The date of birth by itself can't be a great axis that we could retrieve and explain our  data from it so we opted for a conversion from date type to integer

# In[7]:


Fact['Dob']=pd.to_datetime(Fact['Dob'])
date=datetime.today()-Fact['Dob']
Fact['age']=round(date.dt.days/365)


# > ${\textbf{Concatenating firstname and lastname }}$

# In[8]:


Fact["nameDriver"]=Fact["ForeName"]+" "+Fact["SurName"]


# > ${\textbf{ Converting DateTime.Time Format to Milliseconds for a better overlook }}$

# Various methods such as aggregation or summation can't work with dateTime.Time types , also previously we noticed a great deal of null data with is explained in the introduction so nulls will be zeros and time will be converted to milliseconds

# In[9]:


Fact['q1_sec']=0
for i in range(0,len(Fact)):
    t=Fact['q1'].iat[i]
    if t is None:
        Fact['q1_sec'].iat[i] = int(0)
    else:
        Fact['q1_sec'].iat[i] = int(t.hour*3600000000+t.minute*60000000+t.second*1000000+t.microsecond)


# In[10]:


Fact['q2_sec']=0
for i in range(0,len(Fact)):
    t=Fact['q2'].iat[i]
    if t is None:
        Fact['q2_sec'].iat[i] = int(0)
    else:
        Fact['q2_sec'].iat[i] = int(t.hour*3600000000+t.minute*60000000+t.second*1000000+t.microsecond)


# In[11]:


Fact['q3_sec']=0
for i in range(0,len(Fact)):
    t=Fact['q3'].iat[i]
    if t is None:
        Fact['q3_sec'].iat[i] = int(0)
    else:
        Fact['q3_sec'].iat[i] = int(t.hour*3600000000+t.minute*60000000+t.second*1000000+t.microsecond)


# > ${\textbf{Dropping unaffectful columns}}$

# In[12]:


Fact.drop(columns=['raceId',
                   'DriverID',
                   'constructorId',
                   'CircuitId',
                   'dateId',
                   'weatherID',
                   'constructorRef',
                   'DriverRef',
                   'circuitRef',
                   'ForeName',
                   'SurName',
                   'Dob',
                   'number',
                   'time',
                   'Number',
                    'day',
                   'car',
                   'month',
                   'Altitude',
                   'location',
                   'points',
                   'img',
                   'weather',
                   'q1',
                   'q2',
                   'q3'],axis=1,inplace=True)


# > ${\textbf{Columns }}$

# In[13]:


Fact.columns


# > ${\textbf{Data types and shape  }}$

# In[14]:


Fact.info()
Fact.shape


# Overall we have 18 columns and 4759 rows

# > ${\textbf{Head and tails }}$

# In[15]:


Fact.head()


# In[16]:


Fact.tail()


# > ${\textbf{checking of null values }}$

# As previously explained we have anticipated the nulls and fixed them in the preporcessing phase

# In[17]:


Fact.isna().sum()


# In[18]:


selected_columns = Fact[["Code","position","age","q3_sec","driverNationality"]]
df = selected_columns.copy()


# In[19]:


df['winning']=0
for i in range(0,len(df)):
    pos=df['position'].iat[i]
    if pos>0 and pos<6:
         df['winning'].iat[i] = int(1)
    else:
        
         df['winning'].iat[i] = int(0)


# In[20]:


df.drop(columns=['position'],axis=1,inplace=True)


# In[21]:


df1=df
df1=df1.groupby(["Code"]).sum('winning')


# In[22]:


df1["potential"]=df1["winning"]/Fact.groupby(["Code"]).size()


# In[23]:


df=df1["potential"].to_frame()


# In[24]:


df.to_excel(r'potential.xlsx')


# In[25]:


df1 = pd.merge(df,driver, on='Code', how='inner')
df1['Dob']=pd.to_datetime(df1['Dob'])
date=datetime.today()-df1['Dob']
df1['age']=round(date.dt.days/365)
df1.drop(columns=['DriverID','DriverRef','Number','ForeName','SurName','Dob','car'],axis=1,inplace=True)
df1['points'] = df1['points'].fillna(0).astype(np.int64, errors='ignore')
df1['age'] = df1['age'].fillna(0).astype(np.int64, errors='ignore')
df=df1


# In[26]:


df.sort_values(by="potential",ascending=False).head(15)


# In[27]:


le = LabelEncoder()
df["Code"] = le.fit_transform(df["Code"])
df["Nationality"] = le.fit_transform(df["Nationality"])


# In[28]:


df


# In[29]:


X = df.drop(['potential'], axis=1)
y = df[['potential']]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=3)


# In[30]:


# #Régression linéaire multiple (toutes les variables de Boston) 
# linreg2 = LinearRegression(fit_intercept= True, normalize=False) 
# #pour imposer une ordonnée à l'origine (Par défaut ='True')
# # Normalisation des données normalize=True
# linreg2.fit(X_train, y_train)
# y_pred = linreg2.predict(X_test)

# print('test_score = ',linreg2.score(X_test,y_test)) 
# print('R2 = ',r2_score(y_test, y_pred))
# print('MAE = ',mean_squared_error(y_test, y_pred))
# print('RMSE = ', np.sqrt(mean_squared_error(y_test, y_pred)))
# print('MAE = ', mean_absolute_error(y_test, y_pred))
# print('MeadianAE = ', median_absolute_error(y_test, y_pred))

# print('Intercept = ', linreg2.intercept_)
# print('Coefficients : ',linreg2.coef_)

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)


# In[31]:


test={'Code':['0'],'Nationality':['24'],'points':['170'],'age':['30']}
df=pd.DataFrame(test)
pred = regressor.predict(df)
print(pred)


# In[32]:


pickle.dump(regressor, open('performancePredictor.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('performancePredictor.pkl','rb'))

