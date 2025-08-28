import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Downloads\emp_sal.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

# SVM Model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='sigmoid', degree = 5, gamma = 'auto', C=1.0 )
svr_regressor.fit(x,y)



# Visualising the SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x, svr_regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)


#KNN model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=3,weights='distance', leaf_size=30)
knn_reg_model.fit(x,y)

plt.scatter(x, y, color = 'red')
plt.plot(x, knn_reg_model.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

#Decision Tree

from sklearn.tree import DecisionTreeRegressor
dtr_reg_model=DecisionTreeRegressor(criterion='absolute_error',splitter='best',min_samples_split=2 )
dtr_reg_model.fit(x,y)

dtr_reg_pred = dtr_reg_model.predict([[6.5]])
print(dtr_reg_pred)
