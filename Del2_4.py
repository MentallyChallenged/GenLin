import numpy as np
import statsmodels.api as sm
import pandas as pd

# Last inn datasettet
fil = "game_of_thrones_train.csv"
dataframe = pd.read_csv(fil)

# Dummyvariabler for "title" og "house"
title_dummies = pd.get_dummies(dataframe["title"], prefix="title", drop_first=False)
house_dummies = pd.get_dummies(dataframe["house"], prefix="house", drop_first=False)

# Del opp alder i grupper
age_delt = [0, 15, 25, 40, 60, 85, np.inf]
age_label = ["1-15", '16-25', '26-40', '41-60', '61-85', '86+']
dataframe["age_deler"] = pd.cut(dataframe["age"], bins=age_delt, labels=age_label, right=False)

# Lag dummyvariabler for alder
age_dummies = pd.get_dummies(dataframe["age_deler"], prefix="age", drop_first=True)

# Funksjon for å filtrere ut dummyvariabler med færre enn 20 forekomster
def filter_dummies(dummies, threshold=20):
    significant_dummies = []
    for var in dummies.columns:
        if dummies[var].sum() > threshold:
            significant_dummies.append(var)
    return significant_dummies

# Bruk filteret på dummyvariablene
significant_title_dummies = filter_dummies(title_dummies)
significant_house_dummies = filter_dummies(house_dummies)

# Kombiner alle variablene til én DataFrame
X = pd.concat([
    dataframe[["isNoble", "male"]],
    title_dummies[significant_title_dummies],
    house_dummies[significant_house_dummies],
    age_dummies,
    dataframe[["book1", "book2", "book3", "book4", "book5"]]
], axis=1)

# Definer målet (target)
y = dataframe["isAlive"]

# Den endelige modellen med 7 variabler
final_features = [
    'book4', 'age_86+', 'male', 'house_House Targaryen',
    "house_Night's Watch", 'house_House Greyjoy', 'isNoble'
]

# Legg til en konstant (intercept)
X_final = sm.add_constant(X[final_features])

# Fit den endelige modellen
final_model = sm.Logit(y, X_final)
final_result = final_model.fit(method='lbfgs', maxiter=200)

# Print regresjonssammendraget
print(final_result.summary())