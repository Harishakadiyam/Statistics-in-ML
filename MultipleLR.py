import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


# Load the dataset
dataset = pd.read_csv(r'C:\Users\Admin\Downloads\Investment.csv')

space = dataset['DigitalMarketing']
price = dataset['Profit']

X = np.array(space).reshape(-1, 1)
y = np.array(price)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


#  Visualizing the training Test Results 
plt.scatter(X_train, y_train, color= 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()


#Visualizing the Test Results 
plt.scatter(X_test, y_test, color= 'orange')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
plt.show()

#save the trained model to disk
filename = 'Investment_model_pkl'
with open(filename, 'wb')as file:
    pickle.dump(regressor, file)
    
#Load the pickle file 
with open('Investment_model_pkl','rb')as file:
    model = pickle.load(file)
print("model has been pickled and saved as Investment_model_pkl")


