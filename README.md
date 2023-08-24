# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

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
![Screenshot 2023-08-24 083338](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/26a2dfa1-33a1-4b1e-ba7f-90223b14a44c)

![Screenshot 2023-08-24 083348](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/00eb96c6-acf9-49c7-a519-4b4a7dd12e4f)

![Screenshot 2023-08-24 083355](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/8a2078d7-1e05-41e2-9c50-ec0503fdc78f)

![Screenshot 2023-08-24 083406](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/07893f9f-64e4-4bf7-95cb-4427e12c3c7d)

![Screenshot 2023-08-24 083414](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/df032f68-aa8f-48b6-81a3-7b12f50cbbcc)

![Screenshot 2023-08-24 083422](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/63d73c8c-d663-4cc1-b8b4-55a8a159f75d)

![Screenshot 2023-08-24 083429](https://github.com/premalatha-sureshbabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120620842/a6f6fd10-b0a7-443f-ac43-cd1a3ca947a5)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
