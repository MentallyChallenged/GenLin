import numpy as np
import statsmodels.api as sm
import pandas as pd

# Load the dataset
fil = "game_of_thrones_train.csv"
dataframe = pd.read_csv(fil)

# Dummy variables for "title" and "house"
title_dummies = pd.get_dummies(dataframe["title"], prefix="title", drop_first=False)
house_dummies = pd.get_dummies(dataframe["house"], prefix="house", drop_first=False)

# Bin the 'age' column
age_delt = [0, 15, 25, 40, 60, 85, np.inf]
age_label = ["1-15", '16-25', '26-40', '41-60', '61-85', '86+']
dataframe["age_deler"] = pd.cut(dataframe["age"], bins=age_delt, labels=age_label, right=False)

# Create dummy variables for the binned 'age' column
age_dummies = pd.get_dummies(dataframe["age_deler"], prefix="age", drop_first=True)

# Function to filter dummy variables with fewer than 20 occurrences
def filter_dummies(dummies, threshold=20):
    signifikant_dummies = []
    for var in dummies.columns:
        if dummies[var].sum() > threshold:
            signifikant_dummies.append(var)
    return signifikant_dummies

# Apply the filter to title and house dummies
signifikant_title_dummies = filter_dummies(title_dummies)
signifikant_house_dummies = filter_dummies(house_dummies)

# Combine all features into a single DataFrame
X = pd.concat([
    dataframe[["isNoble", "male"]],
    title_dummies[signifikant_title_dummies],
    house_dummies[signifikant_house_dummies],
    age_dummies,
    dataframe[["book1", "book2", "book3", "book4", "book5"]]
], axis=1)

# Selected features for individual logistic regression
valgte_features = ["house_House Frey", "house_House Greyjoy", "house_House Targaryen", "house_Night's Watch",
                  "age_86+", "book4", "isNoble", "male"]

# Initialize an empty list to store results
resultat = []

# Fit individual logistic regression models
for feature in valgte_features:
    # Check if the feature is a dummy variable (house, title, or age)
    if feature.startswith("house_"):
        X_feature = sm.add_constant(house_dummies[feature])  # Use house_dummies
    elif feature.startswith("title_"):
        X_feature = sm.add_constant(title_dummies[feature])  # Use title_dummies
    elif feature.startswith("age_"):
        X_feature = sm.add_constant(age_dummies[feature])  # Use age_dummies
    else:
        X_feature = sm.add_constant(dataframe[feature])  # Use the original dataframe
    
    y = dataframe["isAlive"]
    
    model = sm.Logit(y, X_feature)
    result = model.fit(disp=0)  # Suppress output
    
    # Store the results
    resultat.append({
        "Feature": feature,
        "Coefficient": result.params[1],
        "Std Err": result.bse[1],
        "z-value": result.tvalues[1],
        "p-value": result.pvalues[1],
        "[0.025": result.conf_int().iloc[1, 0],
        "0.975]": result.conf_int().iloc[1, 1]
    })

# Convert results to a DataFrame
resultat_df = pd.DataFrame(resultat)
print(resultat_df)
