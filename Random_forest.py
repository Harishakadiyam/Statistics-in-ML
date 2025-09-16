import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Downloads\emp_sal.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
rfr_reg_model = RandomForestRegressor(n_estimators=6,random_state=0)
rfr_reg_model.fit(x,y)

plt.scatter(x, y, color = 'red')
plt.plot(x, rfr_reg_model.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

rfr_reg_model = rfr_reg_model.predict([[6.5]])
print(rfr_reg_model)