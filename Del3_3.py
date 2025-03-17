import numpy as np
import statsmodels.api as sm
import pandas as pd

train_fil = "game_of_thrones_train.csv"
train_dataframe = pd.read_csv(train_fil)

def prepare_data(dataframe):
    # Dummyvariabler for "house"
    house_dummies = pd.get_dummies(dataframe["house"], prefix="house", drop_first=False)

    age_delt = [0, 15, 25, 40, 60, 85, np.inf]
    age_label = ["1-15", '16-25', '26-40', '41-60', '61-85', '86+']
    dataframe["age_deler"] = pd.cut(dataframe["age"], bins=age_delt, labels=age_label, right=False)

    # Lag dummyvariabler for alder
    age_dummies = pd.get_dummies(dataframe["age_deler"], prefix="age", drop_first=True)

    X = pd.concat([
        dataframe[["isNoble", "male", "book4"]],
        house_dummies,
        age_dummies
    ], axis=1)

    return X

# Forbered treningsdataene
X_train = prepare_data(train_dataframe)
y_train = train_dataframe["isAlive"]

# Den endelige modellen med 7 variabler
final_features = [
    'book4', 'age_86+', 'male', 'house_House Targaryen',
    "house_Night's Watch", 'house_House Greyjoy', 'isNoble'
]

# Legg til en konstant (intercept) til treningsdataene
X_train_final = sm.add_constant(X_train[final_features])

# Tren modellen på treningsdataene
final_model = sm.Logit(y_train, X_train_final)
final_result = final_model.fit(method='lbfgs', maxiter=200)

# Skriv ut koeffisientene
print(final_result.params)

# Formuler den endelige ligningen
coefficients = final_result.params
equation = "log(p/(1-p)) = {:.4f} + {:.4f}*book4 + {:.4f}*age_86+ + {:.4f}*male + {:.4f}*house_House_Targaryen + {:.4f}*house_Night's_Watch + {:.4f}*house_House_Greyjoy + {:.4f}*isNoble".format(
    coefficients['const'], coefficients['book4'], coefficients['age_86+'], coefficients['male'],
    coefficients['house_House Targaryen'], coefficients["house_Night's Watch"], coefficients['house_House Greyjoy'], coefficients['isNoble']
)

print("Den endelige logistiske regresjonsligningen er:")
print(equation)