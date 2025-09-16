import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Downloads\emp_sal.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
dtr_reg_model=DecisionTreeRegressor(criterion='friedman_mse',splitter='random',max_depth=10,min_samples_leaf=1)
dtr_reg_model.fit(x,y)

plt.scatter(x, y, color = 'red')
plt.plot(x, dtr_reg_model.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

dtr_reg_pred = dtr_reg_model.predict([[6.5]])
print(dtr_reg_pred)