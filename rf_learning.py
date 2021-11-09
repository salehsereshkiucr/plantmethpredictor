from sklearn.ensemble import RandomForestClassifier
import numpy as np
import configs as configs
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

config = configs.Arabidopsis_config
organism_name = config['organism_name']
mode = 'Seq'
for at in config['annot_types']:
    mode += at


window_sizes = [100, 200, 400, 800, 1600, 3200]
block_sizes = [(10, 10), (20, 10), (20, 20), (40, 20), (40, 40), (80, 40)]

contexts = ['CG', 'CHG', 'CHH']

test_percent = 0.2
test_val_percent = 0.5

root = 'output_complete/'
i = 1
context = contexts[1]
X = np.load(root + organism_name +'/profiles/' + str(i) + '/X_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
Y = np.load(root + organism_name +'/profiles/' + str(i) + '/Y_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)

window_size = 1000
block_size = block_sizes[0]

def data_preprocess(X, Y, window_size):
    X = np.delete(X, range(4, X.shape[2]), 2)
    b = [j for j in range(int((3200-window_size)/2))] + [j for j in range(3200 - int((3200-window_size)/2), 3200)]
    X = np.delete(X, b, 1)
    X = X.reshape(list(X.shape) + [1])
    #X = np.swapaxes(X, 1, 2)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=None)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_val_percent, random_state=None)
    return x_train, y_train, x_test, y_test, x_val, y_val

x_train, y_train, x_test, y_test, x_val, y_val = data_preprocess(X, Y, window_size)
nsamples, nx, ny, nz = x_train.shape
x_train = x_train.reshape((nsamples, nx*ny))
nsamples, nx, ny, nz = x_test.shape
x_test = x_test.reshape((nsamples, nx*ny))

clf = RandomForestClassifier(random_state=0, n_estimators=500)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))












n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)
