import numpy as np 
import statsmodels as sm 
import pandas as pd
import matplotlib.pylab as plt

fil = "game_of_thrones_train.csv"
data = pd.read_csv(fil)
dataframe = pd.DataFrame(data)

#bruker get dummies fra pandas til å konvertere string data til binæere kategorier 
dataframe_kat = pd.get_dummies(dataframe, columns=["title", "house"], drop_first=True)

age_delt = [0,15,25,40,60,85,np.inf]
age_label = ["1-15",'16-25', '26-40', '41-60', '61-85', '86+']
dataframe["age_deler"] = pd.cut(dataframe["age"], bins = age_delt, labels=age_label, right=False)

variabler = ["isNoble", "male", "age_deler"]
variabler2 = ["book1","book2","book3","book4","book5"]
y = dataframe["isAlive"]

#for å lage dataframen til latex 
def df_to_latex(df, caption, label):
    latex_table = df.to_latex(index=True, escape=False)
    latex_table = f"\\begin{{table}}[h]\n\\centering\n{latex_table}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}"
    return latex_table

kryss_tabell = pd.crosstab(y,
    [dataframe[var] for var in variabler],
    rownames=['isAlive'], 
    colnames=variabler,
    dropna=False,
    margins=True,
    margins_name="Totalt")

kryss_tabell2 = pd.crosstab(y,
    [dataframe[var] for var in variabler2],
    rownames=['isAlive'], 
    colnames=variabler2,
    dropna=False,
    margins=True,
    margins_name="Totalt")


# Print formatted tables
print("Cross-tabulation for isNoble, male, and age_deler:")
print(kryss_tabell.to_string())
print("\nCross-tabulation for books:")
print(kryss_tabell2.to_string())

# Generate LaTeX code
latex_table1 = df_to_latex(kryss_tabell, "Cross-tabulation for isNoble, male, and age_deler", "tab:crosstab1")
latex_table2 = df_to_latex(kryss_tabell2, "Cross-tabulation for books", "tab:crosstab2")

# Print LaTeX code
print("\nLaTeX code for the first table:")
print(latex_table1)
print("\nLaTeX code for the second table:")
print(latex_table2)