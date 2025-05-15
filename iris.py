import pandas as pd 
import numpy as np

df=pd.read_csv("iris.csv")
print(df.head())
print(df.shape)

print(df.isnull().sum())


x=df.iloc[:,:4].values
y=df.iloc[:,4].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

# #feature scaling
# from sklearn.preprocessing import StandardScaler
# ss=StandardScaler()

# x_train=ss.fit_transform(x_train)
# x_test=ss.fit_transform(x_test)

#apply decision tree

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(x_train,y_train)

DecisionTreeClassifier()


y_pred=model.predict(x_test)



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print(cm)

import pickle

pickle.dump(model,open('iris.pkl','wb'))

print("model is created")