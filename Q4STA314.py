import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import statsmodels.api as sm
np.seterr('ignore')

de = pd.read_csv('default.csv')
de.drop(['Unnamed: 0', 'student', 'income'], axis=1, inplace=True)
de['default'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)

LogisticReg = sm.GLM(endog=de['default'], exog=de['balance'], family=sm.families.Binomial()).fit()
print(LogisticReg.summary())


def log_likelihood(par):
    xb = par[0] + par[1]*de['balance']
    return -np.sum((de['default']*xb)-np.log(1+np.exp(xb)))


MLE = minimize(log_likelihood, x0=np.array([0, 0]))
print(MLE.x)




