#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bokeh.plotting import figure, show
import sklearn.naive_bayes as nb
import sklearn.svm as svm

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 600)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None


documentation = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/'

df = pd.read_csv('abalone.csv')
del df['Unnamed: 0']

train, test = train_test_split(df, test_size=0.3)

p = figure(plot_width=500, plot_height=500)

p.inverted_triangle(train[train['sex'] == 'F']['length'], train[train['sex'] == 'F']['whole_weight'], color='red', alpha=0.3)
p.circle(train[train['sex'] == 'M']['length'], train[train['sex'] == 'M']['whole_weight'], color='blue', alpha=0.3)
p.asterisk(train[train['sex'] == 'I']['length'], train[train['sex'] == 'I']['whole_weight'], color='orange', alpha=0.3)

show(p)

clf1 = nb.GaussianNB()
train['feature'] = train[['length', 'whole_weight']].apply(lambda row: np.array([row['length'], row['whole_weight']]), axis=1)
test['feature'] = test[['length', 'whole_weight']].apply(lambda row: np.array([row['length'], row['whole_weight']]), axis=1)

clf1.fit(list(train['feature']), train['sex'])
results = clf1.predict(list(test['feature']))

test['prediction'] = np.array(results)
test['hit'] = test.apply(lambda row: 1 if row['prediction'] == row['sex'] else 0, axis=1)

print('naive bayes has an efficiency of: ', np.round(test.describe().at['mean', 'hit'], 4))


clf2 = svm.SVC(kernel='linear')

clf2.fit(list(train['feature']), train['sex'])
results = clf2.predict(list(test['feature']))

test['prediction'] = np.array(results)
test['hit'] = test.apply(lambda row: 1 if row['prediction'] == row['sex'] else 0, axis=1)

print('SVC linear has an efficiency of: ', np.round(test.describe().at['mean', 'hit'], 4))


clf2 = svm.SVC(kernel='rbf', gamma=1000, C=10)

clf2.fit(list(train['feature']), train['sex'])
results = clf2.predict(list(test['feature']))

test['prediction'] = np.array(results)
test['hit'] = test.apply(lambda row: 1 if row['prediction'] == row['sex'] else 0, axis=1)

print('SVC rbf has an efficiency of: ', np.round(test.describe().at['mean', 'hit'], 4))

