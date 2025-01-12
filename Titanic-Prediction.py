#Titanic-Prediction
This is for predicting the survival of Titanic
#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Importing the Dataset
df= pd.read_csv("C:\Users\sahan\Downloads\archive.zip")
df.shape
#Counting of survival
df['Survived'].value_counts()
#Visualizing the count
sns.countplot(x=df['Survived'], y=df['Pclass'])
<Axes: xlabel='count' , ylabel='Survived'>
df["Gender"]
sns.countplot(x=df['Gender'], y=df['Survived])
<Axes: xlabel='count' , ylabel='Gender">
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df['Gender']=labelencoder.fit_transform(df['Gender'])
df.head()
sns.countplot(x=df['Gender'],y=df['Survived'])
<Axes:xlabel='count' , ylabel='Gender'>
#Droping unnecessary coloumn
df=df.drop(['Age'], axis=1)
x=df['pclass' , 'Gender']
y=df['Survived']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
#LogisticRegression
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=0)
log.fit(x_train,y_train)
#Prediction
pred=print(log.predict(y_test))
print(x_test)
