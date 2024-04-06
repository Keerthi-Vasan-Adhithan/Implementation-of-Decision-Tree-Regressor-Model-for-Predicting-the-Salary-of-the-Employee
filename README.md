# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```py
# Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
# Developed by: KEERTHI VASAN A
# RegisterNumber:  212222240048
```
```py
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
```
```py
data.info()
```
```py
data.isnull().sum()
```
```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
```
```py
x=data[['Position','Level']]
x
```
```py
y=data['Salary']
y
```
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```
```py
from sklearn.tree import DecisionTreeClassifier,plot_tree
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```py
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
```
```py
r2=metrics.r2_score(y_test,y_pred)
r2
```
```py
import matplotlib.pyplot as plt
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```

## Output:
### Printing head

![71](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/fe0a01be-8bfe-4baa-a66e-fee3f274ab9d)

### Printing info about dataset

![72](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/d641d935-cfd5-4bf4-b903-999e2f1930cf)

### Counting the null values

![73](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/6d804d07-4b1a-44f4-a8b6-234f36c42f77)

### Label Encoding the Position Column with 

![74](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/2480f54f-00f9-4329-b4c7-a397cea73e77)

### Spliting the dataset for dependent and independent values

![75](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/1545b693-c92e-4691-b4f4-dca0051a8e76)

![76](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/cf25a1ef-b725-4d5c-91c8-aed2d351c7fa)

### MSE for test_data

![77](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/8f55e236-5262-42a4-a8b8-a9d4625edb5e)

### R2 value for test_data

![78](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/9991b264-765e-4cc1-a6fc-d745e3ec3b92)

### Printing Plot 

![79](https://github.com/Keerthi-Vasan-Adhithan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/107488929/39d3b066-9a6e-4076-a434-4b4ed2dc7899)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
