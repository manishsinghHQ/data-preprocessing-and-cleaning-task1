#Data preprocessing and cleaning
#step 1 import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#step 2 Read dataset
df=pd.read_csv("Titanic-Dataset.csv")
df.head()
#step 3 sanity check of data
df.shape
df.info()
df.isnull().sum()
df.isnull().sum()/df.shape[0]*100
df.duplicated().sum()
for i in df.select_dtypes(include='object').columns:
    print(i)
    print(df[i].value_counts())
    print("***"*10)
df.describe().T
df.describe(include='object').T
# exploratory data analysis
for i in df.select_dtypes(include='object').columns:
    sns.histplot(data=df,x=i)
    plt.show()
s=df.select_dtypes(include='number').columns
for i in s:
  sns.boxplot(data=df,x=i)
  plt.show()
t=df.select_dtypes(include='number').corr()
plt.figure(figsize=(15,15))
sns.heatmap(t,annot=True)
#step 5 handling missing values
df.isnull().sum()
for i in ["Age","Cabin","Embarked"]:
    df[i].fillna(df[i].median(),inplace=True)

df.isnull().sum()
from sklearn.impute import KNNImputer
impute=KNNImputer()
for i in df.select_dtypes(include="number").columns:
            df[i]=impute.fit_transform(df[[i]])
#outlier treatment
def wisker(col):
     q1,q3=np.percentile(col,[25,75])
     iqr=q3-q1
     lw=q1-1.5*iqr
     uw=q3+1.5*iqr
     return lw,uw
wisker(df(['Age']))
#drop dup
df.drop_duplicates()
#encoding of data
pd.get_dummies(data=df.columns=["Sex"])







