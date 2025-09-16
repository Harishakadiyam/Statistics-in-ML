import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Downloads\emp_sal.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

# SVM Model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly', degree = 4, gamma = 'auto', C=1.0 )
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
