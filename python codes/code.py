import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("401K.csv")
# Reading the column "mrate" as X
X = df["mrate"]
# Adding a column of 1's to X
X = sm.add_constant(X)
# Reading the column "prate" as y
y = df["prate"]