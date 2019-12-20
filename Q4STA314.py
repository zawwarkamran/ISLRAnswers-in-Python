import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
np.seterr('ignore')

de = pd.read_csv('default.csv')
de.drop(['Unnamed: 0', 'student', 'income'], axis=1, inplace=True)
de['default'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)

LogisticReg = sm.GLM(endog=de['default'], exog=sm.add_constant(de['balance']), family=sm.families.Binomial()).fit()
print(LogisticReg.summary())


def log_likelihood(par):
    xb = par[0] + par[1]*de['balance']
    return -np.sum((de['default']*xb)-np.log(1+np.exp(xb)))


MLE = minimize(log_likelihood, x0=np.array([0, 0]), method='Nelder-Mead')
print(MLE)

# Bernoulli
values = np.array([1, 1, 1, 0, 0])


def bernloglik(par):
    return -np.sum(values*np.log(par)+(1-values)*np.log(1-par))


MLE_2 = minimize(bernloglik, x0=np.array([0]), method='Nelder-Mead')

parvals = np.linspace(0, 1, 20)
funcvals = list(map(lambda x: -bernloglik(x), parvals))

plt.plot(parvals, funcvals)
plt.show()