import numpy as np
import statsmodels.api as sm
import pandas as pd


fil = "game_of_thrones_train.csv"
dataframe = pd.read_csv(fil)


title_dummies = pd.get_dummies(dataframe["title"], prefix="title", drop_first=False)
house_dummies = pd.get_dummies(dataframe["house"], prefix="house", drop_first=False)

age_delt = [0, 15, 25, 40, 60, 85, np.inf]
age_label = ["1-15", '16-25', '26-40', '41-60', '61-85', '86+']
dataframe["age_deler"] = pd.cut(dataframe["age"], bins=age_delt, labels=age_label, right=False)


age_dummies = pd.get_dummies(dataframe["age_deler"], prefix="age", drop_first=True)

def filter_dummies(dummies, threshold=20):
    significant_dummies = []
    for var in dummies.columns:
        if dummies[var].sum() > threshold:
            significant_dummies.append(var)
    return significant_dummies


significant_title_dummies = filter_dummies(title_dummies)
significant_house_dummies = filter_dummies(house_dummies)


X = pd.concat([
    dataframe[["isNoble", "male"]],
    title_dummies[significant_title_dummies],
    house_dummies[significant_house_dummies],
    age_dummies,
    dataframe[["book1", "book2", "book3", "book4", "book5"]]
], axis=1)

y = dataframe["isAlive"]

selected_features = [
    "house_House Frey", "house_House Greyjoy", "house_House Targaryen", "house_Night's Watch",
    "age_86+", "book4", "isNoble", "male"
]

# Starte en tom moddell
included_features = []
results = []

# Stegvis forover-analyse
while True:
    best_aic = np.inf
    best_feature = None
    best_model = None

    # Tester hver gjenværende variabel
    for feature in selected_features:
        if feature not in included_features:
            # Lag en midlertidig modell med den nye variabelen
            temp_features = included_features + [feature]
            X_temp = sm.add_constant(X[temp_features])
            model = sm.Logit(y, X_temp)
            result = model.fit(method='lbfgs', maxiter=200, disp=0)

            # Sjekk om AIC er bedre (her om det blir lavere)
            if result.aic < best_aic:
                best_aic = result.aic
                best_feature = feature
                best_model = result

    # Hvis ingen variabler forbedrer AIC, stopp
    if best_feature is None:
        break

    # Legger til den beste variabelen i modellene
    included_features.append(best_feature)
    selected_features.remove(best_feature)

    # Lagre resultatene
    results.append({
        "Modell": f"Modell med {len(included_features)} variabler",
        "Inkluderte Variabler": included_features.copy(),
        "Devians": -2 * best_model.llf,
        "Antall Parametre": len(best_model.params),
        "AIC": best_model.aic
    })

# Konverter resultatene til en dataFrame
results_df = pd.DataFrame(results)
print(results_df)