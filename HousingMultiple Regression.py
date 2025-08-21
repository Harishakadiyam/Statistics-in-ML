import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Admin\Downloads\18th- SLR\15th- SLR\SLR - House price prediction\House_data.csv")

X = dataset.iloc[:,-1 :]
y = dataset.iloc[:, 2].values
