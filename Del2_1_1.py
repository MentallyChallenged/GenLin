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
    significant_dummies = []
    for var in dummies.columns:
        if dummies[var].sum() > threshold:
            significant_dummies.append(var)
    return significant_dummies

# Apply the filter to title and house dummies
significant_title_dummies = filter_dummies(title_dummies)
significant_house_dummies = filter_dummies(house_dummies)

# Combine all features into a single DataFrame
X = pd.concat([
    dataframe[["isNoble", "male"]],
    title_dummies[significant_title_dummies],
    house_dummies[significant_house_dummies],
    age_dummies,
    dataframe[["book1", "book2", "book3", "book4", "book5"]]
], axis=1)

# Selected features for the model
selected_features = [
    "house_House Frey", "house_House Greyjoy", "house_House Targaryen", "house_Night's Watch",
    "age_86+", "book4", "isNoble", "male"
]

# Add a constant (intercept) to the features
X_selected = sm.add_constant(X[selected_features])

# Define the target variable
y = dataframe["isAlive"]

# Fit the logistic regression model using L-BFGS
model = sm.Logit(y, X_selected)
result = model.fit(method='lbfgs', maxiter=200)

# Print the summary
print(result.summary())