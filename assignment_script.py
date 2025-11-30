# import libraries
import pandas as pd
import pymc as pm
import numpy as np

#import data
data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv")

# split into control/test groups
control = data[data["version"] == "gate_30"]
test = data[data["version"] == "gate_40"]