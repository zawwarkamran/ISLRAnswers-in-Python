import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('auto.csv', sep=',')

df.drop('Unnamed: 0', axis=1, inplace=True)
df['mpg01'] = df['mpg'].apply(lambda x: 1 if x > df['mpg'].median() else 0)

# Perform a correlation of the continuous variables
print(df[['mpg', 'displacement', 'acceleration', 'horsepower']].corr())

df_cleaned = df.drop(['cylinders', 'year', 'origin', 'name', 'acceleration'], axis=1)

X = df_cleaned.drop(['mpg01', 'mpg'], axis=1)
y = df_cleaned['mpg01']
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
y_train = y_train.astype('int')
y_test = y_test.values.reshape(-1, 1)

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
print('Confusion Matrix for LDA:')
print(confusion_matrix(y_true=y_test, y_pred=lda_model.predict(x_test)))
print(classification_report(y_test, lda_model.predict(x_test)))

qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
print('Confusion Matrix for QDA:')
print(confusion_matrix(y_test, qda_model.predict(x_test)))
print(classification_report(y_test, qda_model.predict(x_test)))

log_model = LogisticRegression(solver='liblinear')
log_model.fit(X_train, y_train)
print('Confusion Matrix for Logistic Regression:')
print(confusion_matrix(y_test, log_model.predict(x_test)))
print(classification_report(y_test, log_model.predict(x_test)))

lst = []
KNN_levels = [1, 5, 10, 15, 20, 30, 50, 100, 150, 200]
for k in KNN_levels:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    lst.append((k, knn_model.fit(X_train, y_train)))

print('Confusion Matrix for KNN starting at 1 through 200:')
for item in lst:
    print('Number of Neighbors:', item[0])
    print(confusion_matrix(y_test, item[1].predict(x_test)))
    print(classification_report(y_test, item[1].predict(x_test)))
