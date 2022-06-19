# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import packages.
2. Read the csv file for datas.
3. Check for duplicate data in the given data. 
4. Using sklearn transform every column.
5. Assign a column to x.
6. Train the model using test datas.
7. Using logistic gradient predict the datas .
8. End.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vijay Ganesh N
RegisterNumber:  212221040177
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
datasets=pd.read_csv('/content/Social_Network_Ads (1).csv')
X=datasets.iloc[:,[2,3]].values 
Y=datasets.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_X
X_Train=sc_X.fit_transform(X_Train)
X_Test=sc_X.transform(X_Test)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)
Y_pred=classifier.predict(X_Test)
Y_pred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_Test,Y_pred)
cm
from sklearn import metrics
accuracy=metrics.accuracy_score(Y_Test,Y_pred)
accuracy
recall_sensitivity = metrics.recall_score(Y_Test,Y_pred,pos_label=1)
recall_specificity = metrics.recall_score(Y_Test,Y_pred,pos_label=0)
recall_sensitivity ,recall_specificity
from matplotlib.colors import ListedColormap
X_Set,Y_Set=X_Train,Y_Train
X1,X2=np.meshgrid(np.arange(start=X_Set[:,0].min()-1,stop=X_Set[:,0].max()+1,step=0.01),np.arange(start=X_Set[:,1].min()-1,stop=X_Set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X2.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_Set)):
  plt.scatter(X_Set[Y_Set==j,0],X_Set[Y_Set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression(Training Set)')
plt.xlabel('age')
plt.ylabel('estimated Salary')
plt.legend()
plt.show()

```

## Output:
#### Logistic Regression using Gradient Descent
![logistic regression using gradient descent](https://github.com/vijayganeshn96/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/blob/main/Screenshot%202022-06-19%20170712.png)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

