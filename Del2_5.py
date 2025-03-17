import numpy as np
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

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

final_features = [
    'book4', 'age_86+', 'male', 'house_House Targaryen',
    "house_Night's Watch", 'house_House Greyjoy', 'isNoble'
]

X_final = sm.add_constant(X[final_features])

final_model = sm.Logit(y, X_final)
final_result = final_model.fit(method='lbfgs', maxiter=200)

# Predikerte sannsynligheter
predicted_probabilities = final_result.predict(X_final)

# Rå residualer
raw_residuals = y - predicted_probabilities

# Standardiserte residualer
standardized_residuals = raw_residuals / np.sqrt(predicted_probabilities * (1 - predicted_probabilities))

# Beregner devianse residualer
def calculate_deviance_residuals(y, predicted_probabilities):
    deviance_residuals = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        if y[i] == 1:
            deviance_residuals[i] = np.sqrt(-2 * np.log(predicted_probabilities[i]))
        else:
            deviance_residuals[i] = -np.sqrt(-2 * np.log(1 - predicted_probabilities[i]))
    return deviance_residuals

deviance_residuals = calculate_deviance_residuals(y, predicted_probabilities)

# Residualplot for rå residualer
plt.figure(figsize=(10, 6))
plt.scatter(predicted_probabilities, raw_residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikerte sannsynligheter")
plt.ylabel("Rå residualer")
plt.title("Residualplot for rå residualer")
plt.show()

# Residualplot for standardiserte residualer
plt.figure(figsize=(10, 6))
plt.scatter(predicted_probabilities, standardized_residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikerte sannsynligheter")
plt.ylabel("Standardiserte residualer")
plt.title("Residualplot for standardiserte residualer")
plt.show()

# Residualplot for Deviance Residualer
plt.figure(figsize=(10, 6))
plt.scatter(predicted_probabilities, deviance_residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predikerte sannsynligheter")
plt.ylabel("Deviance Residualer")
plt.title("Residualplot for Deviance Residualer")
plt.show()

# QQ-plot for standardiserte residualer
plt.figure(figsize=(10, 6))
stats.probplot(standardized_residuals, dist="norm", plot=plt)
plt.title("QQ-plot for standardiserte residualer")
plt.show()