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

# for å lage individuell kryss_tabell for alle variabelene 
for var in variabler:
    kryss_tabell = pd.crosstab(y, dataframe[var],
        rownames=['isAlive'], 
        colnames=[var],
        dropna=False,
        margins=True,
        margins_name="Totalt")
    
    print(f"\nCross-tabulation for {var}:")
    print(kryss_tabell.to_string())
    
    latex_table = df_to_latex(kryss_tabell, f"Cross-tabulation for {var}", f"tab:crosstab_{var}")
    print(f"\nLaTeX code for {var} table:")
    print(latex_table)


for var in variabler2:
    kryss_tabell = pd.crosstab(y, dataframe[var],
        rownames=['isAlive'], 
        colnames=[var],
        dropna=False,
        margins=True,
        margins_name="Totalt")
    
    print(f"\nCross-tabulation for {var}:")
    print(kryss_tabell.to_string())
    
    latex_table = df_to_latex(kryss_tabell, f"Cross-tabulation for {var}", f"tab:crosstab_{var}")
    print(f"\nLaTeX code for {var} table:")
    print(latex_table)
    
# Extract dummy variable columns for "title" and "house"
title_dummies = [col for col in dataframe_kat.columns if col.startswith('title_')]
house_dummies = [col for col in dataframe_kat.columns if col.startswith('house_')]

# Function to create cross-tabulations for variables with more than 20 occurrences
def create_crosstabs_for_significant_vars(dummies):
    for var in dummies:
        # Convert boolean values to numeric
        data_numeric = dataframe_kat[var].astype(int)
        # Check if the number of occurrences of the value 1 is over 20
        if data_numeric.sum() > 20:
            kryss_tabell = pd.crosstab(y, dataframe_kat[var],
                rownames=['isAlive'], 
                colnames=[var],
                dropna=False,
                margins=True,
                margins_name="Totalt")
            
            print(f"\nCross-tabulation for {var}:")
            print(kryss_tabell.to_string())
            
            latex_table = df_to_latex(kryss_tabell, f"Cross-tabulation for {var}", f"tab:crosstab_{var}")
            print(f"\nLaTeX code for {var} table:")
            print(latex_table)

# Create cross-tabulations for significant title and house dummy variables
create_crosstabs_for_significant_vars(title_dummies)
create_crosstabs_for_significant_vars(house_dummies)