import numpy as np 
import statsmodels as sm 
import pandas as pd
import matplotlib.pylab as plt

fil = "game_of_thrones_train.csv"
data = pd.read_csv(fil)
dataframe = pd.DataFrame(data)

latex_output = dataframe.describe().style.format(precision=3).to_latex()
print(latex_output)


