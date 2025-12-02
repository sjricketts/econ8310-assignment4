# import libraries
import pandas as pd
import pymc as pm
import numpy as np

#import data
data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv")

# split into control/test groups
control = data[data["version"] == "gate_30"]
test = data[data["version"] == "gate_40"]

#split by retention day
control_1 = control["retention_1"].values
control_7 = control["retention_7"].values
test_1 = test["retention_1"].values
test_7 = test["retention_7"].values 

# PyMC model
with pm.Model() as model:
  # priors that are not known
  p_control = pm.Beta("p_control", alpha=1, beta=1)
  p_test = pm.Beta("p_test", alpha=1, beta=1)

with model:
  # use Bernoulli for probabilities
  # maybe should be binomial instead???
  obs_control = pm.Bernoulli("obs_control",p=p_control, observed=control_1)
  obs_test = pm.Bernoulli("obs_test",p=p_test, observed=test_1)

