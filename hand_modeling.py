import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import dataframe_image as dfi
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import pickle

def Standardization(Series):
    mean_series = Series.mean()
    std_series = Series.std()
    Series = (Series - mean_series) / std_series
    return Series

def LowerColumns(dataframe):
    for i in dataframe.columns:
        dataframe.rename(columns = {f"{i}": f"{i.lower()}"}, inplace=True)

def DropFeatures(df):
    df = df.drop(columns=['score', 'hand'])
    return df

def ColReplace(col):
    df.loc[df[col].str.contains('scissor'), 'move'] = 'scissor'
    df.loc[df[col].str.contains('paper'), 'move'] = 'paper'
    df.loc[df[col].str.contains('rock'), 'move'] = 'rock'

def CleanDataframe(df):
    df['move'] = df['move'].replace({"scissor": 0, "paper": 1, "rock": 2})
    df['hand'] = df['hand'].replace({"Right": 1, "Left": 0})
    df = df[df['score'] > 0.95]
    return df


def SplitXY(df):
    y = df['move']
    x = df.drop(columns=['move'])
    y_tr, y_ts, x_tr, x_ts = train_test_split(y, x, test_size=.3, random_state=False)
    return y_tr, y_ts, x_tr, x_ts


def evaluate_classification_model(y_train, y_pred_train, y_test, y_pred_test):
    performance_df = pd.DataFrame({'Error_metric': ['Accuracy','Precision','Recall'],
                                   'Train': [accuracy_score(y_train, y_pred_train),
                                             precision_score(y_train, y_pred_train, average = 'weighted'),
                                             recall_score(y_train, y_pred_train, average = 'weighted')],
                                   'Test': [accuracy_score(y_test, y_pred_test),
                                            precision_score(y_test, y_pred_test, average = 'weighted'),
                                            recall_score(y_test, y_pred_test, average = 'weighted')]})

    pd.options.display.float_format = '{:.2f}'.format

    df_train = pd.DataFrame({'Real': y_train, 'Predicted': y_pred_train})
    df_test  = pd.DataFrame({'Real': y_test,  'Predicted': y_pred_test})

    return performance_df, df_train, df_test


#%% PREPARE DATAFRAME
df = pd.read_csv("data/hands_coords.csv")
#fig_path = 'C:/Users/tomma/Documents/data_science/berlin/final_project/presentation/'

# clean dataframe with previously defined functions
ColReplace('move')
LowerColumns(df)
df = CleanDataframe(df)
df = DropFeatures(df)
df = df.dropna()


## to train the model, I only use the distances between the finger landmarks, and drop the signle coordinates
df2 = df.iloc[:, -10:len(df.columns)]

## split test train
y_tr, y_ts, x_tr, x_ts = SplitXY(df2)
columns_names = x_tr.columns
print(x_tr.columns, len(x_tr.columns))


##% NOW TRAIN THE MODELS
##% 1) DECISION TREES
tree = DecisionTreeClassifier(random_state=12345)
tree.fit(x_tr, y_tr)

## prediction
y_pred_train_tree = tree.predict(x_tr)
y_pred_test_tree  = tree.predict(x_ts)

## accuracy
trees_acc = classification_report(y_ts, y_pred_test_tree)
print(trees_acc)

## feature importance
tree_importance = pd.DataFrame(tree.feature_importances_)
tree_importance.index = columns_names
tree_importance = tree_importance.sort_values(by=0, ascending=False)
tree_importance = tree_importance.head(10)
dfi.export(tree_importance, "img/tree_feat.png")
print("DECISION TREE/n", tree_importance.head(10))


## error metrics
y_pred_train_tree = tree.predict(x_tr)
y_pred_test_tree = tree.predict(x_ts)

error_metrics_tree, y_train_vs_predicted, \
y_test_vs_predicted = evaluate_classification_model(y_tr, y_pred_train_tree,
                                                    y_ts, y_pred_test_tree)


## export table
#error_metrics_tree = error_metrics_tree.style.background_gradient() #adding a gradient based on values in cell
dfi.export(error_metrics_tree, "img/tree_table.png")
print('done')


#%% 2) KNN CLASSIFIER
knn = KNeighborsClassifier(n_neighbors=15) # n_neighbors = K
knn.fit(x_tr, y_tr)

## prediction
y_pred_train_knn = knn.predict(x_tr)
y_pred_test_knn  = knn.predict(x_ts)

## accuracy
knn_acc = classification_report(y_ts, y_pred_test_knn)
print(knn_acc)

## feature importance
knn_res = permutation_importance(knn, x_tr, y_tr, scoring='accuracy')
knn_importance = pd.DataFrame(knn_res.importances_mean)

try:
    knn_importance.index = columns_names
    knn_importance = knn_importance.sort_values(by=0, ascending=False)
    knn_importance = knn_importance.head(10)
    dfi.export(knn_importance, "img/knn_feat.png")
    print("KNN CLASSIFIER/n", knn_importance.head(10))

except:
    pass

## error metrics
y_pred_train_knn = knn.predict(x_tr)
y_pred_test_knn = knn.predict(x_ts)

error_metrics_knn, y_train_vs_predicted, \
y_test_vs_predicted = evaluate_classification_model(y_tr, y_pred_train_knn,
                                                    y_ts, y_pred_test_knn)



## export table
#error_metrics_knn = error_metrics_knn.style.background_gradient() #adding a gradient based on values in cell
dfi.export(error_metrics_knn, "img/knn_table.png")
print('ok')


#%% 3) RANDOM FOREST WITH PARAMETER TUNING
'''
rfc = RandomForestClassifier()
param_grid = {
    'n_estimators': [500, 750, 1000],
    'min_samples_split': [2, 3],
    #'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    ##'max_samples' : ['None', 0.5],
    'max_depth':[3,5,10]
    ## 'bootstrap':[True,False]
}
grid_search = GridSearchCV(rfc, param_grid, cv=5,return_train_score=True,n_jobs=-1,)
grid_search

grid_search.fit(x_tr, y_tr)

grid_search.best_params_ #To check the best set of parameters returned

pd.DataFrame(grid_search.cv_results_)

print('ok')
'''
#%%
#rfc = RandomForestClassifier()
rfc = RandomForestClassifier(max_depth=10,
                             #min_samples_leaf=20,
                             min_samples_split = 2,
                             max_features = 'sqrt',
                             n_estimators = 1000,
                             #bootstrap = True,
                             #oob_score = True,
                             random_state = 0)

cross_val_scores = cross_val_score(rfc, x_tr, y_tr, cv=5)
rfc.fit(x_tr, y_tr)

y_pred_train_rfc = rfc.predict(x_tr)
y_pred_test_rfc = rfc.predict(x_ts)

## error metrics
y_pred_train_rfc = rfc.predict(x_tr)
y_pred_test_rfc = rfc.predict(x_ts)

error_metrics_rfc, y_train_vs_predicted, \
y_test_vs_predicted = evaluate_classification_model(y_tr, y_pred_train_rfc,
                                                    y_ts, y_pred_test_rfc)
error_metrics_rfc

## feature importance
rfc_importance = pd.DataFrame(rfc.feature_importances_)
rfc_importance.index = columns_names
rfc_importance = rfc_importance.sort_values(by=0, ascending=False)
rfc_importance = rfc_importance.head(10)
dfi.export(rfc_importance, "img/rfc_feat.png")
print("RANDOM FOREST/n", rfc_importance.head(10))

## export table
#error_metrics_knn = error_metrics_rfc.style.background_gradient() #adding a gradient based on values in cell
dfi.export(error_metrics_rfc, "img/rfc_table.png")
print('ok')


#%% plot trees
from sklearn.tree import plot_tree

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(34,20))
plot_tree(tree, max_depth=3, filled=True, rounded=True,
          feature_names=x_tr.columns, fontsize=15)
plt.show()



#%% 4) SGD CLASSIFIER
sgd = SGDClassifier(random_state=12345)
sgd.fit(x_tr, y_tr)

## prediction
y_pred_train_sgd = sgd.predict(x_tr)
y_pred_test_sgd = sgd.predict(x_ts)

## accuracy
sgd_acc = classification_report(y_ts, y_pred_test_sgd)
print(sgd_acc)



## error metrics
error_metrics_sgd, y_train_vs_predicted, \
y_test_vs_predicted = evaluate_classification_model(y_tr, y_pred_train_sgd,
                                                    y_ts, y_pred_test_sgd)


## export table
#error_metrics_sgd = error_metrics_sgd.style.background_gradient() #adding a gradient based on values in cell
dfi.export(error_metrics_sgd, "img/sgd_table.png")
print('ok')



#%% 5) SVM
svc = svm.SVC(random_state=12345)
svc.fit(x_tr, y_tr)

## prediction
y_pred_train_svc = svc.predict(x_tr)
y_pred_test_svc  = svc.predict(x_ts)

## accuracy
svc_acc = classification_report(y_ts, y_pred_test_svc)
print(svc_acc)

## error metrics
y_pred_train_svc = svc.predict(x_tr)
y_pred_test_svc = svc.predict(x_ts)

error_metrics_svc, y_train_vs_predicted, \
y_test_vs_predicted = evaluate_classification_model(y_tr, y_pred_train_svc,
                                                    y_ts, y_pred_test_svc)


## export table
#error_metrics_svc = error_metrics_svc.style.background_gradient() #adding a gradient based on values in cell
dfi.export(error_metrics_svc, "img/svc_table.png")
print('ok')


#%%
# TREE
fig, ax = plt.subplots(1,2, figsize=(14,8))
color_map = "BuGn"
plot_confusion_matrix(tree, x_tr, y_tr,ax=ax[0], values_format = 'd', cmap=color_map)
ax[0].title.set_text("Train Set")

plot_confusion_matrix(tree, x_ts, y_ts,ax=ax[1],values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")
fig.suptitle('DECISION TREES')

plt.savefig('img/tree_CM.png')


#%% PLOTS
# KNN
fig, ax = plt.subplots(1,2, figsize=(14,8))
color_map = "BuGn"
plot_confusion_matrix(knn, x_tr, y_tr,ax=ax[0], values_format = 'd', cmap=color_map)
ax[0].title.set_text("Train Set")

plot_confusion_matrix(knn, x_ts, y_ts,ax=ax[1],values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")
fig.suptitle('KNN')

plt.savefig('img/knn_CM.png')


#%% RF PLOTS
## confusion matrix
fig, ax = plt.subplots(1,2, figsize=(14,8))
color_map = "BuGn"

plot_confusion_matrix(rfc, x_tr, y_tr,ax=ax[0], values_format = 'd', cmap=color_map)
ax[0].title.set_text("Train Set")

plot_confusion_matrix(rfc, x_ts, y_ts,ax=ax[1], values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")
fig.suptitle('RANDOM FOREST')

plt.savefig('img/rfc_CM.png')


#%%
# SGD
fig, ax = plt.subplots(1,2, figsize=(14,8))
color_map = "BuGn"
plot_confusion_matrix(sgd, x_tr, y_tr,ax=ax[0], values_format = 'd', cmap=color_map)
ax[0].title.set_text("Train Set")

plot_confusion_matrix(sgd, x_ts, y_ts,ax=ax[1],values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")
fig.suptitle('SGD')

plt.savefig('img/sgd_CM.png')


#%%
# SVC
fig, ax = plt.subplots(1,2, figsize=(14,8))
color_map = "BuGn"
plot_confusion_matrix(svc, x_tr, y_tr,ax=ax[0], values_format = 'd', cmap=color_map)
ax[0].title.set_text("Train Set")

plot_confusion_matrix(svc, x_ts, y_ts,ax=ax[1],values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")
fig.suptitle('svc')

plt.savefig('img/svc_CM.png')


#%%
### SAVE MODEL
import pickle
with open("models/knn.pickle", "wb") as f:
    pickle.dump(knn,f)

with open(r"models/tree.pickle", "wb") as f:
    pickle.dump(tree,f)

with open(r"models/rfc.pickle", "wb") as f:
    pickle.dump(rfc,f)

with open(r"models/sgd.pickle", "wb") as f:
    pickle.dump(sgd,f)

with open(r"models/svc.pickle", "wb") as f:
    pickle.dump(svc,f)


print('end')

#%%
