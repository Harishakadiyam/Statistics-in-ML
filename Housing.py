import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
dataset = pd.read_csv(r'C:\Users\Admin\Downloads\18th- SLR\15th- SLR\SLR - House price prediction\House_data.csv')
space=dataset['sqft_living']
price=dataset['price']

x = np.array(space).reshape(-1,1)
y = np.array('price')

#split the data into Train and Test
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=5, random_state=0)

#fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain) 
                
#predicting the prices
pred = regressor.predict(xtest)

#visualizing the training Test Results
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Training Dataset")
plt.xlabel("space")
plt.ylabel("price")
plt.show()

#visualizing the Test Results
plt.scatter(xtest, ytest, color = 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test Dataset")
plt.xlabel("space")
plt.ylabel("price")
plt.show()



#save the trained model to disk
filename = 'house_price_model_pkl'
with open(filename, 'wb')as file:
    pickle.dump(regressor, file)
    
#Load the pickle file 
with open('house_price_model_pkl','rb')as file:
    loaded_model = pickle.load(file)
print("model has been pickled and saved as house_price_model.pkl")