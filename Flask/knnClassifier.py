#!/usr/bin/env python
# coding: utf-8

# #### Import

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pip
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
weather= pd.read_sql('select * from "WeatherDim"', con=DbConnect())
penalty= pd.read_sql('select * from "PenalityDim"', con=DbConnect())
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
races.rename(columns = {'weather_fk':'weatherID'}, inplace = True)
result.rename(columns = {'penalty_fk':'penalityId'}, inplace = True)


# In[5]:


df1 = pd.merge(races,date, on='dateId', how='inner')
Racedim = pd.merge(df1,circuit, on='CircuitId', how='inner')
df2 = pd.merge(result,Racedim, on='raceId', how='inner')
df3 = pd.merge(df2,driver, on='DriverID', how='inner')
df4 = pd.merge(df3,constructor, on='constructorId', how='inner')
df5 = pd.merge(df4,status, on='statusId', how='inner')
df6 = pd.merge(df5,weather, on='weatherID', how='inner')
df7 = pd.merge(df6,penalty, on='penalityId', how='inner')


# ### Data Anlysis

# ##### Preparing

# In[6]:


Fact=df7


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
                   'statusId',
                   'statusId',
                   'weatherID',
                   'penalityId',
                   'dateId_x',
                   'dateId_y',
                   'constructorRef',
                   'DriverRef',
                   'circuitRef',
                   'ForeName',
                   'SurName',
                   'Dob',
                   'Number',
                   'Code',
                   'day',
                   'month',
                   'raceName',
                   'driverCode'],axis=1,inplace=True)


# In[11]:


Fact.rename(columns = {'name_x':'nameGP','name_y':'nameCircuit','points':'penalityPoints'}, inplace = True)


# In[12]:


Fact.info()


# In[13]:


Fact.shape


# In[14]:


Fact.head()


# In[15]:


Fact.tail()


# #####  Checking for null Data

# In[16]:


Fact.isnull().sum()*100/len(Fact)


# In[17]:


Fact['weather'].fillna(Fact['weather'].mode().iloc[0], inplace=True)
Fact['fastest_lapspeed'].fillna(Fact['fastest_lapspeed'].median(), inplace=True)
Fact['laptime_sec']=0
for i in range(0,len(Fact)):
    t=Fact['laptime'].iat[i]
    if t is None:
        Fact['laptime_sec'].iat[i] = int(0)
    else:
        Fact['laptime_sec'].iat[i] = int(t.hour*3600000000+t.minute*60000000+t.second*1000000+t.microsecond)


# In[18]:


# heatmap

plt.figure(figsize=(17,12))
sns.heatmap(Fact.corr(),annot=True)
plt.show()


# ## Algorithms

# driver's performance in RS

# Relevent data

# In[19]:


DriverPRS=Fact.drop(columns=[
                   'date',
                   'img',
                   'time',
                   'nameCircuit',
                    'location',
                    'country',
                    'Nationality',
                    'constructorName',
                    'constructorNationality',
                    'reason',
                    'laptime',
'longitude',
'penalityPoints',
'status',
'Altitude',
'nameGP',
'pitstop',
'points_x',
'wins',
'round',
'year',
'points_y',
'car',
'weather',
'nameDriver','latitude'])


# In[20]:


DriverPRS.head()


# ##### Variables encoding

# from sklearn.preprocessing import LabelEncoder
# lb_make = LabelEncoder()
# DriverPRS["nameDriver"] = lb_make.fit_transform(DriverPRS["nameDriver"])
# DriverPRS["car"] = lb_make.fit_transform(DriverPRS["car"])
# DriverPRS["weather"] = lb_make.fit_transform(DriverPRS["weather"])
# DriverPRS

# ##### Correlation test

# In[21]:


# heatmap

plt.figure(figsize=(17,12))
sns.heatmap(DriverPRS.corr(),annot=True)
plt.show()


# In[22]:


DriverPRS['position_perf']=0
for i in range(0,len(DriverPRS)):
    pos=DriverPRS['rank'].iat[i]
    if pos>0 and pos<6:
         DriverPRS['position_perf'].iat[i] = int(1)
    elif pos>5 and pos<11:
         DriverPRS['position_perf'].iat[i] = int(2)
    elif pos>10 and pos<16:
         DriverPRS['position_perf'].iat[i] = int(3)
    else:
         DriverPRS['position_perf'].iat[i] = int(4)


# In[23]:


DriverPRS


# In[24]:


#separate the other attributes from the predicting attribute
X = DriverPRS.drop(columns=['position_perf','rank'])
#separte the predicting attribute into Y for model training
y = DriverPRS['position_perf']


# In[25]:


# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=5)


# In[26]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train);


# In[27]:


print('training accuracy = ' + str(dt.score(X_train, y_train)))
print('test accuracy = '+ str(dt.score(X_test, y_test)))


# In[28]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
val_score=[]
K=np.arange(1,50)
for i in K:
    score=cross_val_score(KNeighborsClassifier(n_neighbors=i),X_train,y_train,cv=5)
    val_score.append(score.mean())

val_score


# In[29]:


print(len(val_score))


# In[34]:


# chargement du modèle KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 
# Entrainer le modèle avec la méthode fit
KNN=KNeighborsClassifier(n_neighbors=2) 
KNN.fit(X_train,y_train)


# In[35]:


# Calculate models' score
y_pred=KNN.predict(X_test)
print(KNN.score(X_train,y_train))
print(KNN.score(X_test, y_test))


# In[47]:


test={'laps':['77'],'fastest_lapspeed':['160'],'age':['27'],'laptime_sec':['64006428000']}
df=pd.DataFrame(test)
pred = KNN.predict(df)
print(pred)


# In[50]:


import pickle
pickle.dump(KNN, open('Knn.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('Knn.pkl','rb'))

