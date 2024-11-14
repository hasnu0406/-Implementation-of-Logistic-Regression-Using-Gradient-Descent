# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and remove unnecessary columns (`sl_no` and `salary`).
2. Convert relevant columns to categorical data types, then encode them as numerical codes.
3. Separate the dataset into features (`X`) and target (`Y`).
4. Initialize random weights (`theta`) for the model.
5. Define a sigmoid function for logistic regression and a loss function to calculate the model error.
6. Implement gradient descent to iteratively update `theta` based on the learning rate and the gradient.
7. Train the model by running gradient descent for a set number of iterations.
8. Define a prediction function that uses `theta` to classify new data.
9. Calculate and print the model's accuracy by comparing predictions to actual values in `Y`.
10. Predict and print the model's output for two new data samples (`xnew`).

## Program:
```Python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: HASNA MUBARAK AZEEM
RegisterNumber: 212223240052
*/

import pandas as pd
import numpy as np
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y = Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:
![image](https://github.com/user-attachments/assets/bcbd07a4-5db1-4b60-adc1-a08d006947b7)
![image](https://github.com/user-attachments/assets/0fe1eca4-f905-43c6-8576-992adb60821e)
![image](https://github.com/user-attachments/assets/bf916798-b2dc-460a-af7b-4cd21d87dc90)
![image](https://github.com/user-attachments/assets/645a3e16-7eb4-4ffe-9c84-90bc5914c1cd)
![image](https://github.com/user-attachments/assets/83136d53-d407-43db-8a3c-332e24a97cc6)
![image](https://github.com/user-attachments/assets/e2eb7a0e-db68-4daa-8b74-d03f18d6c71c)
![image](https://github.com/user-attachments/assets/d8ab2cce-8846-495e-8b64-8be4c5e244f9)

![image](https://github.com/user-attachments/assets/4ca395d9-9237-4a74-a00e-9c8370c9e05a)

![image](https://github.com/user-attachments/assets/8595325f-8230-476c-842c-642af389a781)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

