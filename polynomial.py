import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Downloads\emp_sal.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Linear Regression visualization
plt.scatter(x, y, color= 'red') #real salary data
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
lin_reg_5 = LinearRegression()
lin_reg_5.fit(x_poly,y)


plt.scatter(x, y, color= 'red') 
plt.plot(x, lin_reg_5.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict([[6.5]])

poly_model_pred= lin_reg_5.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)



