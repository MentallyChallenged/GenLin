import numpy as np 
import statsmodels.api as sm 
import pandas as pd
import matplotlib.pylab as plt

fil = "game_of_thrones_train.csv"
data = pd.read_csv(fil)
dataframe = pd.DataFrame(data)

#dummy variabler for "title" og "house"
title_dummies = pd.get_dummies(dataframe["title"], prefix="title", drop_first=False)
house_dummies = pd.get_dummies(dataframe["house"], prefix="house", drop_first=False)

#deler opp alder inni i de gruppene 
age_delt = [0,15,25,40,60,85,np.inf]
age_label = ["1-15",'16-25', '26-40', '41-60', '61-85', '86+']
dataframe["age_deler"] = pd.cut(dataframe["age"], bins = age_delt, labels=age_label, right=False)

age_dummies = pd.get_dummies(dataframe["age_deler"], prefix="age", drop_first=True)

#for filtre ut alle forekomninger av tittel og house som er under 20 tilfeller
def filter_dummies(dummies, threshold=20):
    signifikant_dummies = []
    for var in dummies.columns:
        if dummies[var].sum() > threshold:
            signifikant_dummies.append(var)
    return signifikant_dummies

#bruke filtret på dummiesene mine:
signifikant_title_dummies = filter_dummies(title_dummies)
signifikant_house_dummies = filter_dummies(house_dummies)

# Legge sammen alle variablene sammen, slik at jeg kan tilpasse modell 
X = pd.concat([dataframe[["isNoble", "male"]],
               title_dummies[signifikant_title_dummies],
               house_dummies[signifikant_house_dummies],
               age_dummies,
               dataframe[["book1", "book2", "book3", "book4", "book5"]]
               ], axis=1)

X = sm.add_constant(X)
y = dataframe["isAlive"]
X = X.drop(columns=["title_Archmaester"])

model= sm.Logit(y,X)
resultat = model.fit(method='lbfgs', maxiter=200)
print(resultat.summary())