# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Jaisree B
RegisterNumber:  212224230100
*/
```
```
      import pandas as pd
      from sklearn.preprocessing import LabelEncoder
      from sklearn.model_selection import train_test_split
      from sklearn.tree import DecisionTreeRegressor
      from sklearn import metrics
      
      # Load the data
      data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\sem 3\ml\Salary.csv")
      print(data.head())
      
      # Info and null value check
      data.info()
      print(data.isnull().sum())
      
      # Encode categorical variable
      le = LabelEncoder()
      data["Position"] = le.fit_transform(data["Position"])
      print(data.head())
      
      # Feature and target selection
      x = data[["Position", "Level"]]
      y = data["Salary"]
      
      # Train/test split
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
      
      # Decision Tree Regressor
      dt = DecisionTreeRegressor()
      dt.fit(x_train, y_train)
      y_pred = dt.predict(x_test)
      
      # Metrics
      mse = metrics.mean_squared_error(y_test, y_pred)
      print("Mean Squared Error:", mse)
      
      r2 = metrics.r2_score(y_test, y_pred)
      print("R2 Score:", r2)
      
      # Predict for new sample (use DataFrame for correct feature names)
      sample = pd.DataFrame([[5, 6]], columns=["Position", "Level"])
      prediction = dt.predict(sample)
      print("Prediction for [5,6]:", prediction)
      



```



## Output:


<img width="580" height="625" alt="Screenshot 2025-10-15 152723" src="https://github.com/user-attachments/assets/16092879-a50e-4907-a014-116b852db74d" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
