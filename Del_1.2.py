import numpy as np 
import statsmodels as sm 
import pandas as pd
import matplotlib.pylab as plt

fil = "game_of_thrones_train.csv"
data = pd.read_csv(fil)
dataframe = pd.DataFrame(data)

#bruker get dummies fra pandas til å konvertere string data til binæere kategorier 
dataframe_kat = pd.get_dummies(dataframe, columns=["title", "house"], drop_first=True)
#print(f"One-Hot Encoded Data using Pandas:\n{dataframe_kat}\n")

variabler = ["isNoble", "male", "age"]

for var in variabler:
    plt.figure(figsize=(8,6))
    plt.hist(dataframe_kat[var], bins=20, color='red', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram av {var}', fontsize = 14)
    plt.xlabel(var, fontsize=12)
    plt.ylabel('Antall tilfeller', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

antall_his = 0
for var in dataframe_kat.columns:
    if var.startswith('title_') or var.startswith('house_'):
        # Konverter boolske verdier til numeriske så de kan histograferes
        data_numeric = dataframe_kat[var].astype(int)
        # Sjekker om antall tilfeller av veriden 1 er over 20
        if data_numeric.sum() > 20:
            antall_his +=1
            plt.figure(figsize=(8, 6))
            plt.hist(data_numeric, bins=20, color='red', edgecolor='black', alpha=0.7)
            plt.title(f'Histogram av {var}', fontsize=14)
            plt.xlabel(var, fontsize=12)
            plt.ylabel('Antall tilfeller', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
print(antall_his)

variabler2 = ["book1","book2","book3","book4","book5"]
for var in variabler2:
    plt.figure(figsize=(8,6))
    plt.hist(dataframe_kat[var], bins=20, color='red', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram av {var}', fontsize = 14)
    plt.xlabel(var, fontsize=12)
    plt.ylabel('Antall tilfeller', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()