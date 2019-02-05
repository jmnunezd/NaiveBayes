#%%
from sklearn.model_selection import train_test_split
from bokeh.plotting import figure, show
import sklearn.naive_bayes as nb
from contour import *

data_set_metadata = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/'

df = pd.read_csv('abalone.csv')
del df['Unnamed: 0']
df['adult'] = df['sex'].apply(lambda sex: {'I': 0, 'M': 1, 'F': 1}.get(sex, ' '))
df['feature'] = df[['length', 'whole_weight']].apply(lambda row: np.array([row['length'], row['whole_weight']]), axis=1)

x = list(df['feature'])
y = list(df['adult'])

models = (svm.SVC(kernel='linear', C=1),
          svm.SVC(kernel='rbf', C=20),
          svm.SVC(kernel='rbf', gamma=20, C=1),
          svm.SVC(kernel='rbf', gamma=20, C=20))
models = (clf.fit(x, y) for clf in models)

# title for the plots
titles = ('Linear C=1',
          'rbf C=20',
          'rbf gamma=20 and C=1',
          'rbf gamma=20 and C=20')


# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

x0, x1 = df['length'], df['whole_weight']
xx, yy = make_meshgrid(x0, x1)

train, test = train_test_split(df, test_size=0.2)

# bokeh plot for the train set.
p = figure(plot_width=500, plot_height=500)
p.circle(train[train['adult'] == 1]['length'], train[train['adult'] == 1]['whole_weight'], color='blue', alpha=0.3)
p.asterisk(train[train['adult'] == 0]['length'], train[train['adult'] == 0]['whole_weight'], color='red', alpha=0.3)

# show(p)


for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x0, x1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Abalon Length')
    ax.set_ylabel('Abalone Whole Weight')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)


plt.show()



# clf1 = nb.GaussianNB()
# clf1.fit(list(train['feature']), train['adult'])
# results = clf1.predict(list(test['feature']))
#
# test['prediction'] = np.array(results)
# test['hit'] = test.apply(lambda row: 1 if row['prediction'] == row['adult'] else 0, axis=1)
#
# print('naive bayes has an efficiency of: ', np.round(test.describe().at['mean', 'hit'], 4))
#
#
# clf2 = svm.SVC(kernel='linear')
#
# clf2.fit(list(train['feature']), train['sex'])
# results = clf2.predict(list(test['feature']))
#
# test['prediction'] = np.array(results)
# test['hit'] = test.apply(lambda row: 1 if row['prediction'] == row['sex'] else 0, axis=1)
#
# print('SVC linear has an efficiency of: ', np.round(test.describe().at['mean', 'hit'], 4))
#
#
# clf2 = svm.SVC(kernel='rbf', gamma=1000, C=10)
#
# clf2.fit(list(train['feature']), train['sex'])
# results = clf2.predict(list(test['feature']))
#
# test['prediction'] = np.array(results)
# test['hit'] = test.apply(lambda row: 1 if row['prediction'] == row['sex'] else 0, axis=1)
#
# print('SVC rbf has an efficiency of: ', np.round(test.describe().at['mean', 'hit'], 4))




