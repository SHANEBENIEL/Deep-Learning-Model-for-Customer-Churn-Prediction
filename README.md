# Deep-Learning-Model-for-Customer-Churn-Prediction
#I have used tensorflow to create an simple customer churn prediction model using Kaggle DataSet where i used techniques such data cleaning, data exploration and feature extraction and all other preprocess
#techniques except balancing the dataset.
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.colab import files
uploaded=files.upload()
pd.set_option('display.max_columns',None)
df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.drop('customerID',axis=1,inplace=True)
df[pd.to_numeric(df.TotalCharges,errors='coerce').isna()]
df.drop(df[pd.to_numeric(df['TotalCharges'],errors='coerce').isnull()].index,axis=0,inplace=True)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'])
tenure_churn_no=df[df['Churn']=='No'].tenure
tenure_churn_yes=df[df['Churn']=='Yes'].tenure
plt.hist([tenure_churn_yes,tenure_churn_no])
#to clear understanding we need to modify the code little bit
plt.hist([tenure_churn_yes,tenure_churn_no],color=['green','red'],label=['Churn=Yes','Churn=No'])
#first attempt label is not worked so we need add this code of line
plt.legend()
plt.xlabel('Tenure')
plt.ylabel('Number of customers')
plt.title('Tenure VS Churn')
mc_churn_no=df[df.Churn=='No'].MonthlyCharges
mc_churn_yes=df[df.Churn=='Yes'].MonthlyCharges
plt.hist([mc_churn_yes,mc_churn_no],color=['orange','blue'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.xlabel('Monthly Charges')
plt.ylabel('Number of Customers')
plt.title('Monthlly Charges VS Churn')
def unique_col_values(DataFrame):
  for column in DataFrame:
    #if DataFrame[column].dtypes=='object':
      print(f"{column}:{DataFrame[column].unique()}")

unique_col_values(df)
#here ther is also lot of no internet services but this is ame as no so we
#have to replace them with no
df.replace('No internet service','No',inplace=True)
#also  want to replace no phone service
df.replace('No phone service','No',inplace=True)
unique_col_values(df)
#Next we have to convert text to numbers (0-1) so the machine will understand
yes_no_columns=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
  df[col].replace({'Yes':1,'No':0},inplace=True)
for col in df:
  print(f"{col}:{df[col].unique()}")
df['gender'].replace({'Female':0,'Male':1},inplace=True)
for col in df:
  print(f"{col}:{df[col].unique()}")
df_columns=['InternetService','Contract','PaymentMethod']
for col in df_columns:
  df=pd.get_dummies(data=df,columns=[col],dtype=int)
#another way of doing this
#df=pd.get_dummies(data=df,columns=['InternetService','Contract','PaymentMethod'])
df.head()
col_to_scale=['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[col_to_scale]=scaler.fit_transform(df[col_to_scale])
df.head(5)
x=df.drop('Churn',axis=1)
y=df.Churn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
model=keras.Sequential([
    keras.layers.Dense(64,input_shape=(26,),activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')

])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)
model.evaluate(x_test,y_test)
y_pred=model.predict(x_test)
y_pred[15:20]
y_test[11:20]
y_pre=[]
for i in y_pred:
  if i>0.5:
    y_pre.append(1)
  else:
    y_pre.append(0)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pre))
import seaborn as sb
sb.heatmap(confusion_matrix(y_test,y_pre),annot=True,fmt='d',cmap='Blues')
#fmt is used to avoid this kind of values like 1.23e-12
plt.xlabel('Predicted')
plt.ylabel('Truth')
