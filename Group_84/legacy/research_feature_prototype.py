# RESEARCH CODE - exported as-is from notebooks/research_prototype.ipynb
# This file is the *before* state for Objective 1.2 and 1.4. It is deliberately
# preserved unformatted and unlinted so the improvement is measurable.
# DO NOT run black/isort on this file - it is evidence, not shipping code.
import pandas as pd, numpy as np
import os, sys, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/analyst/Desktop/loan_data_processed.csv")   # hardcoded local path
print(df.shape)
print( df.head() )

# quick look
tmp = df.describe()
X = df.drop('LoanApproved',axis=1)
y = df['LoanApproved']

# feature ideas -- try a few, keep whatever looks good
df['LoanToIncomeRatio'] = df['LoanAmount']/df['AnnualIncome']       # NOTE: blows up when income is 0
df['SavingsToLoanRatio'] = df['SavingsAccountBalance']/df['LoanAmount']
df['ratio2'] = df['MonthlyDebtPayments']*12/df['AnnualIncome']
df['x'] = df['CreditScore']/850
FEATURES=['Age','AnnualIncome','CreditScore','LoanAmount','LoanToIncomeRatio','SavingsToLoanRatio','ratio2','x','DebtToIncomeRatio','CreditCardUtilizationRate','BankruptcyHistory','PreviousLoanDefaults']

X_train,X_test,y_train,y_test=train_test_split(df[FEATURES],y,test_size=0.2)      # no random_state -> not reproducible
m = RandomForestClassifier()
m.fit(X_train,y_train)
p=m.predict(X_test)
print("acc",accuracy_score(y_test,p))

def get_ratio(a,b) :
    return a/b        # no zero guard, no types, no docstring

# plot importances
imp = m.feature_importances_
plt.barh(FEATURES,imp) ; plt.show()

# TODO: clean this up before the demo
# TODO: why does the score move every run?
try:
    m.predict(pd.DataFrame())
except:
    pass                # bare except swallows everything
