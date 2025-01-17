# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

  1. Import the standard Libraries.
   
  2.Set variables for assigning dataset values.

  3.Import linear regression from sklearn.

  4.Assign the points for representing in the graph.

  5.Predict the regression for marks by using the representation of the graph.

  6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.Prema Latha
RegisterNumber: 212222230112
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
```

## Output:
## Head

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/e1f04b6f-732c-4f43-9f87-f842802d2b49)

## Tail

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/e55cecf2-5242-47c9-b2bc-b8f35becc8da)

## Array value of x

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/3651fe4e-399c-4556-acbd-fa50ec653b12)

## Array value of y

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/6d05cd2d-7b96-4b59-a638-d729bbe9d503)

## Values of Y prediction

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/88297343-72ea-4073-ab9e-8890fc26e3b4)

## Array values of Y test

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/5481cc59-898c-430d-8c89-e399ef66faa0)

## Training set graph

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/9bbd2eea-ac72-462f-92e1-530c29425d26)

## Test set graph

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/8e417bdd-2f52-48e6-95dc-f249c5e7c826)

## Values of MSE,MAE,RMSE

![image](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/5811a394-1b50-4a7b-bb87-aed705c9a95d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
