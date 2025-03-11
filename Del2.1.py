import numpy as np 
import statsmodels as sm 
import pandas as pd
import matplotlib.pylab as plt

fil = "game_of_thrones_train.csv"
data = pd.read_csv(fil)
dataframe = pd.DataFrame(data)

#bruker get dummies fra pandas til å konvertere string data til binæere kategorier 
dataframe_kat = pd.get_dummies(dataframe, columns=["title", "house"], drop_first=True)

age_delt = [0,15,25,40,60,85,np.inf]
age_label = ["1-15",'16-25', '26-40', '41-60', '61-85', '86+']
dataframe["age_deler"] = pd.cut(dataframe["age"], bins = age_delt, labels=age_label, right=False)

variabler = ["isNoble", "male", "age_deler"]
variabler2 = ["book1","book2","book3","book4","book5"]
y = dataframe["isAlive"]

dataframe.head()

""" #split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
"""