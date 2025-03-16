import numpy as np
import statsmodels.api as sm
import pandas as pd

# Last inn treningsdataene
train_fil = "game_of_thrones_train.csv"
train_dataframe = pd.read_csv(train_fil)

# Last inn testdataene
test_fil = "game_of_thrones_test.csv"
test_dataframe = pd.read_csv(test_fil)

# Funksjon for å forberede dataene (dummyvariabler, aldersgrupper, etc.)
def prepare_data(dataframe, train_columns=None):
    # Dummyvariabler for "title" og "house"
    title_dummies = pd.get_dummies(dataframe["title"], prefix="title", drop_first=False)
    house_dummies = pd.get_dummies(dataframe["house"], prefix="house", drop_first=False)

    # Del opp alder i grupper
    age_delt = [0, 15, 25, 40, 60, 85, np.inf]
    age_label = ["1-15", '16-25', '26-40', '41-60', '61-85', '86+']
    dataframe["age_deler"] = pd.cut(dataframe["age"], bins=age_delt, labels=age_label, right=False)

    # Lag dummyvariabler for alder
    age_dummies = pd.get_dummies(dataframe["age_deler"], prefix="age", drop_first=True)

    # Kombiner alle variablene til en DataFrame
    X = pd.concat([
        dataframe[["isNoble", "male"]],
        title_dummies,
        house_dummies,
        age_dummies,
        dataframe[["book1", "book2", "book3", "book4", "book5"]]
    ], axis=1)

    # Hvis train_columns er oppgitt, sørg for at X har de samme kolonnene
    if train_columns is not None:
        missing_columns = set(train_columns) - set(X.columns)
        for col in missing_columns:
            X[col] = 0  # Fyll manglende kolonner med 0
        X = X[train_columns]  # Sorter kolonnene i riktig rekkefølge

    return X

# Forbered treningsdataene
X_train = prepare_data(train_dataframe)
y_train = train_dataframe["isAlive"]

# Lagre kolonnenavnene fra treningsdataene
train_columns = X_train.columns

# Forbered testdataene (uten 'isAlive')
X_test = prepare_data(test_dataframe, train_columns)

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

# Legg til en konstant (intercept) til testdataene
X_test_final = sm.add_constant(X_test[final_features])

# Predikerte sannsynligheter på testdataene
predicted_probabilities_test = final_result.predict(X_test_final)

# Konverter predikerte sannsynligheter til binære prediksjoner
y_pred_test = (predicted_probabilities_test >= 0.5).astype(int)

# Lagre prediksjonene i testdataene (valgfritt)
test_dataframe["Predicted_isAlive"] = y_pred_test

# Inkluder alle forklaringsvariabler og prediksjoner
output = test_dataframe.copy()
output["Predicted_isAlive"] = y_pred_test

# Sjekk at alle kolonnene i final_features finnes i output
for feature in final_features:
    if feature not in output.columns:
        output[feature] = 0  # Fyll manglende kolonner med 0

# Velg kun de 7 funksjonene og prediksjonen for å lagre i CSV
output_final = output[final_features + ["Predicted_isAlive"]]

# Skriv ut resultatene
print(output_final)
output_final.to_csv("prediksjoner.csv", index=False)
output_final.to_excel("prediksjoner.xlsx", index=False)