import numpy as np 
import statsmodels as sm 
import pandas as pd
import matplotlib.pylab as plt

fil = "game_of_thrones_train.csv"
data = pd.read_csv(fil)
dataframe = pd.DataFrame(data)

latex_output = dataframe.describe().style.format(precision=3).to_latex()
print(latex_output)

variables = ["title", "house", "noble", "age", "male"]

for var in variables:
    plt.figure(figsize=(8, 6))
    plt.hist(dataframe[var], bins=20, color='red', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram av {var}', fontsize=14)
    plt.xlabel(var, fontsize=12)
    plt.ylabel('Antall tilfeller', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


