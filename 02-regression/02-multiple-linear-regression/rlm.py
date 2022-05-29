
#Regresion lineal multiple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y, exog = x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x

def backwardElimination_ajustado(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y, exog = x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x



df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Variables Dummy
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#print(y_pred)
#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Construir el modelo optimo  RLM utilizando la eliminacion hacia atras
#X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]#.tolist()

#X = sm.add_constant(X).tolist()

SL = 0.05
#results = sm.OLS(endog = y, exog = X).fit()
#max_var = max(results.pvalues).astype(float)
#print(results.pvalues)

X_opt = sm.add_constant(X)
X_Modeled = backwardElimination(X_opt, SL)

#print(results.summary())
# regression_ols = sm.OLS(endog = y, exog = X_opt).fit()
# print(regression_ols.sumary())

#print(X)
# print(y)
