import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', 30)

boston_data = pd.read_csv('Boston.csv', sep=',')
boston_data.drop('Unnamed: 0', axis=1, inplace=True)

# Lets perform some analysis first
sns.pairplot(boston_data.corr())
plt.show()
# Coolwarm color map is the easiest to quickly notice correlations
sns.heatmap(boston_data.corr().round(1), annot=True, cmap='coolwarm')
plt.show()
print(boston_data.corr().round(2))

# Create a binary variable where if the crim value is below the median crim value; it is assigned to 0, otherwise 1.
boston_data['Crim01'] = boston_data['crim'].apply(lambda x: 1 if x > boston_data['crim'].median() else 0)

cov_matrix = boston_data.drop('Crim01', axis=1).cov()
print(np.linalg.det(cov_matrix))



