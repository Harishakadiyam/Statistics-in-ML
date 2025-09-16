import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Downloads\emp_sal.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg_model.fit(x,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

# Visualising the SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x,knn_reg_model.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)