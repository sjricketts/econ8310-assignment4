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


# function for 1 day retention
def ab_1(obs_control, obs_test):
  # PyMC model
  with pm.Model() as model:

    # priors that are not known
    p_control = pm.Beta("p_control", alpha=1, beta=1)
    p_test = pm.Beta("p_test", alpha=1, beta=1)

    # Bernoulli probabilities
    obs_control = pm.Bernoulli("obs_control",p=p_control, observed=control_1)
    obs_test = pm.Bernoulli("obs_test",p=p_test, observed=test_1)

    # differences
    diff_raw = pm.Deterministic("diff_raw", p_test - p_control)
    diff_percent = pm.Deterministic("diff_percent", (p_test - p_control) / p_control)

    # samples
    trace = pm.sample(draws=2000, tune=2000, target_accept=0.95, random_seed=311, return_inferencedata=False)

  return model, trace

# call function for 1 day rentention
retention1_mod, retention1_trace = ab_1(control_1, test_1)

# print results
print("control: ", retention1_trace["p_control"].mean())
print("test: ", retention1_trace["p_test"].mean())
print("raw difference: ",retention1_trace["diff_raw"].mean())
print("percent difference: ",retention1_trace["diff_percent"].mean()*100)


# function for 7 day retention
def ab_7(obs_control, obs_test):
  # PyMC model
  with pm.Model() as model:

    # priors that are not known
    p_control = pm.Beta("p_control", alpha=1, beta=1)
    p_test = pm.Beta("p_test", alpha=1, beta=1)

    # Bernoulli probabilities
    obs_control = pm.Bernoulli("obs_control",p=p_control, observed=control_7)
    obs_test = pm.Bernoulli("obs_test",p=p_test, observed=test_7)

    # differences
    diff_raw = pm.Deterministic("diff_raw", p_test - p_control)
    diff_percent = pm.Deterministic("diff_percent", (p_test - p_control) / p_control)

    # samples
    trace = pm.sample(draws=2000, tune=2000, target_accept=0.95, random_seed=311, return_inferencedata=False)

  return model, trace


# call function for 7 day rentention
retention7_mod, retention7_trace = ab_7(control_7, test_7)

print("Retention for 7 Day\n")
print("control: ", retention7_trace["p_control"].mean())
print("test: ", retention7_trace["p_test"].mean())
print("raw difference: ",retention7_trace["diff_raw"].mean())
print("percent difference: ",retention7_trace["diff_percent"].mean()*100)