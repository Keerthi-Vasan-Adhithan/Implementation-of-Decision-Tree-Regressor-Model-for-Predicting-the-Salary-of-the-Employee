# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### step 1. start the program
#### step 2. Import pandas
#### step 3. Import Decision tree classifier
#### step 4. Fit the data in the model
#### step 5. Find the accuracy score
#### step 6. Stop the program


## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KEERTHI VASAN A
RegisterNumber:  212222240048
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred) 
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])


```

## Output:
#### data.head()
![Screenshot 2023-06-03 182518](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115707860/74928a7e-5490-4f21-a455-081786ea5ce3)

#### data.info()
![Screenshot 2023-06-03 182529](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115707860/ca791c21-ecb7-4869-95a4-3b833cc925e3)

#### isnull() and sum()
![Screenshot 2023-06-03 182535](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115707860/e349e8a7-c2f3-4afe-9707-603ee307cd5b)

#### data.head() for salary 
![Screenshot 2023-06-03 182544](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115707860/fa80dab8-37c3-44c8-86ac-9301b2128636)

#### MSE value
![Screenshot 2023-06-03 182650](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115707860/d48a86c5-919c-47f2-8ffd-8ca51ca8687c)

#### r2 value
![Screenshot 2023-06-03 182656](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115707860/c9c2d65d-f9ff-4a10-9a8e-7270c067d025)

#### data prediction
![image](https://github.com/Jaiganesh235/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118657189/485e2f7f-5060-40f2-8ae2-931a8cd39867)

 
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
