
# coding: utf-8

# In[ ]:


# Based on https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the train and test dataset, which is in the "../input/" directory
train = pd.read_csv("../input/train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("../input/test.csv") # the train dataset is now a Pandas DataFrame

# Let's see what's in the trainings data - Jupyter notebooks print the result of the last thing you do
train.head()

# Press shift+enter to execute this cell


# In[ ]:


# happy customers have TARGET==0, unhappy custormers have TARGET==1
# A little less then 4% are unhappy => unbalanced dataset
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df


# # var3: nationality of the customer

# In[ ]:


# Top-10 most common values
train.var3.value_counts()[:10]


# In[ ]:


# 116 values in column var3 are -999999
# var3 is suspected to be the nationality of the customer
# -999999 would mean that the nationality of the customer is unknown
train.loc[train.var3==-999999].shape


# In[ ]:


# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
train = train.replace(-999999,2)
train.loc[train.var3==-999999].shape


# # Add feature that counts the number of zeros in a row

# In[ ]:


X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']


# # num_var4 : number of bank products

# In[ ]:


# According to dmi3kno (see https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223)
# num_var4 is the number of products. Let's plot the distribution:
train.num_var4.hist(bins=100)
plt.xlabel('Number of bank products')
plt.ylabel('Number of customers in train')
plt.title('Most customers have 1 product with the bank')
plt.show()


# In[ ]:


# Let's look at the density of the of happy/unhappy customers in function of the number of bank products
sns.FacetGrid(train, hue="TARGET", size=6)    .map(plt.hist, "num_var4")    .add_legend()
plt.title('Unhappy cosutomers have less products')
plt.show()


# In[ ]:


train[train.TARGET==1].num_var4.hist(bins=6)
plt.title('Amount of unhappy customers in function of the number of products');


# # Var38
# var38 is important according to XGBOOST
# see https://www.kaggle.com/cast42/santander-customer-satisfaction/xgboost-with-early-stopping/files
# 
# Also RFC thinks var38 is important
# see https://www.kaggle.com/tks0123456789/santander-customer-satisfaction/data-exploration/notebook
# 
# Var38 is suspected to be the mortage value with the bank. If the mortage is with another bank the national
# average is used. 
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19895/var38-is-mortgage-value
# 
# [dmi3kno](https://www.kaggle.com/dmi3kno) says that var38 is value of the customer: [https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223](https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223)

# In[ ]:


train.var38.describe()


# In[ ]:


# How is var38 looking when customer is unhappy ?
train.loc[train['TARGET']==1, 'var38'].describe()


# In[ ]:


# Histogram for var 38 is not normal distributed
train.var38.hist(bins=1000);


# In[ ]:


train.var38.map(np.log).hist(bins=1000);


# In[ ]:


# where is the spike between 11 and 12  in the log plot ?
train.var38.map(np.log).mode()


# In[ ]:


# What are the most common values for var38 ?
train.var38.value_counts()


# the value 117310.979016 appears 14868 times in colum var38

# In[ ]:


# the most common value is very close to the mean of the other values
train.var38[train['var38'] != 117310.979016494].mean()


# In[ ]:


# what if we exclude the most common value
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()


# In[ ]:


# Look at the distribution
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100);


# In[ ]:


# Above plot suggest we split up var38 into two variables
# var38mc == 1 when var38 has the most common value and 0 otherwise
# logvar38 is log transformed feature when var38mc is 0, zero otherwise
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0


# In[ ]:


#Check for nan's
print('Number of nan in var38mc', train['var38mc'].isnull().sum())
print('Number of nan in logvar38',train['logvar38'].isnull().sum())


# # var15

# The most important feature for XGBoost is var15. According to [a Kaggle form post](https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/110414#post110414)
#     var15 is the age of the customer. Let's explore var15

# In[ ]:


train['var15'].describe()


# In[ ]:


#Looks more normal, plot the histogram
train['var15'].hist(bins=100);


# In[ ]:


# Let's look at the density of the age of happy/unhappy customers
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend()
plt.title('Unhappy customers are slightly older');


# # saldo_var30

# In[ ]:


train.saldo_var30.hist(bins=100)
plt.xlim(0, train.saldo_var30.max());


# In[ ]:


# improve the plot by making the x axis logarithmic
train['log_saldo_var30'] = train.saldo_var30.map(np.log)


# In[ ]:


# Let's look at the density of the age of happy/unhappy customers for saldo_var30
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "log_saldo_var30")    .add_legend();


# # Explore the interaction between var15 (age) and var38

# In[ ]:


sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "var38", "var15")    .add_legend();


# In[ ]:


sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120]); # Age must be positive ;-)


# In[ ]:


# Exclude most common value for var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120]);


# In[ ]:


# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend();


# In[ ]:


# What is density of n0 ?
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "n0")    .add_legend()
plt.title('Unhappy customers have a lot of features that are zero');


# # Select the most important features

# In[ ]:


from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# First select features based on chi2 and f_classif
p = 3

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)


# In[ ]:


# Make a dataframe with the selected features and the target variable
X_sel = train[features+['TARGET']]


# # var36

# In[ ]:


X_sel['var36'].value_counts()


# var36 is most of the times 99 or [0,1,2,3]

# In[ ]:


# Let's plot the density in function of the target variabele
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var36")    .add_legend()
plt.title('If var36 is 0,1,2 or 3 => less unhappy customers');


# In above plot we see that the density of unhappy custormers is lower when var36 is not 99

# In[ ]:


# var36 in function of var38 (most common value excluded) 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend();


# Let's seperate that in two plots

# In[ ]:


sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend()
plt.title('If var36==0, only happy customers');


# In[ ]:


# Let's plot the density in function of the target variabele, when var36 = 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6)    .map(sns.kdeplot, "logvar38")    .add_legend();


# # num_var5

# In[ ]:


train.num_var5.value_counts()


# In[ ]:


train[train.TARGET==1].num_var5.value_counts()


# In[ ]:


train[train.TARGET==0].num_var5.value_counts()


# In[ ]:


sns.FacetGrid(train, hue="TARGET", size=6)    .map(plt.hist, "num_var5")    .add_legend();


# In[ ]:


sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "num_var5")    .add_legend();


# In[ ]:


sns.pairplot(train[['var15','var36','logvar38','TARGET']], hue="TARGET", size=2, diag_kind="kde");


# In[ ]:


train[['var15','var36','logvar38','TARGET']].boxplot(by="TARGET", figsize=(12, 6));


# In[ ]:


# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(train[['var15','var36','logvar38','TARGET']], "TARGET");


# # now look at all 8 features together

# In[ ]:


features


# In[ ]:


radviz(train[features+['TARGET']], "TARGET");


# In[ ]:


sns.pairplot(train[features+['TARGET']], hue="TARGET", size=2, diag_kind="kde");


# # Correlations

# In[ ]:


cor_mat = X.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax);


# In[ ]:


cor_mat = X_sel.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax);


# In[ ]:


# only important correlations and not auto-correlations
threshold = 0.7
important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0])     .unstack().dropna().to_dict()
unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['attribute pair', 'correlation'])
# sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['correlation']).argsort()[::-1]]
unique_important_corrs


# # Clusters 

# In[ ]:


# Recipe from https://github.com/mgalardini/python_plotting_snippets/blob/master/notebooks/clusters.ipynb
import matplotlib.patches as patches
from scipy.cluster import hierarchy
from scipy.stats.mstats import mquantiles
from scipy.cluster.hierarchy import dendrogram, linkage


# In[ ]:


# Correlate the data
# also precompute the linkage
# so we can pick up the 
# hierarchical thresholds beforehand

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# scale to mean 0, variance 1
train_std = pd.DataFrame(scale(X_sel))
train_std.columns = X_sel.columns
m = train_std.corr()
l = linkage(m, 'ward')


# In[ ]:


# Plot the clustermap
# Save the returned object for further plotting
mclust = sns.clustermap(m,
               linewidths=0,
               cmap=plt.get_cmap('RdBu'),
               vmax=1,
               vmin=-1,
               figsize=(14, 14),
               row_linkage=l,
               col_linkage=l)


# In[ ]:


# Threshold 1: median of the
# distance thresholds computed by scipy
t = np.median(hierarchy.maxdists(l))


# In[ ]:


# Plot the clustermap
# Save the returned object for further plotting
mclust = sns.clustermap(m,
               linewidths=0,
               cmap=plt.get_cmap('RdBu'),
               vmax=1,
               vmin=-1,
               figsize=(12, 12),
               row_linkage=l,
               col_linkage=l)

# Draw the threshold lines
mclust.ax_col_dendrogram.hlines(t,
                               0,
                               m.shape[0]*10,
                               colors='r',
                               linewidths=2,
                               zorder=1)
mclust.ax_row_dendrogram.vlines(t,
                               0,
                               m.shape[0]*10,
                               colors='r',
                               linewidths=2,
                               zorder=1)

# Extract the clusters
clusters = hierarchy.fcluster(l, t, 'distance')
for c in set(clusters):
    # Retrieve the position in the clustered matrix
    index = [x for x in range(m.shape[0])
             if mclust.data2d.columns[x] in m.index[clusters == c]]
    # No singletons, please
    if len(index) == 1:
        continue

    # Draw a rectangle around the cluster
    mclust.ax_heatmap.add_patch(
        patches.Rectangle(
            (min(index),
             m.shape[0] - max(index) - 1),
                len(index),
                len(index),
                facecolor='none',
                edgecolor='r',
                lw=3)
        )

plt.title('Cluster matrix')

pass


# For clustering with more features, have a look at: [https://www.kaggle.com/cast42/santander-customer-satisfaction/correlation-pairs](https://www.kaggle.com/cast42/santander-customer-satisfaction/correlation-pairs)



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,2)


# Replace 9999999999 with NaN
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/111360#post111360
# training = training.replace(9999999999, np.nan)
# training.dropna(inplace=True)
# Leads to validation_0-auc:0.839577

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)
# # Add log of var38
# X['logvar38'] = X['var38'].map(np.log1p)
# # Encode var36 as category
# X['var36'] = X['var36'].astype('category')
# X = pd.get_dummies(X)

# Add PCA components as features
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X_normalized = normalize(X, axis=0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
X['PCA1'] = X_pca[:,0]
X['PCA2'] = X_pca[:,1]

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

p = 86 # 308 features validation_1-auc:0.848039
p = 80 # 284 features validation_1-auc:0.848414
p = 77 # 267 features validation_1-auc:0.848000
p = 75 # 261 features validation_1-auc:0.848642
# p = 73 # 257 features validation_1-auc:0.848338
# p = 70 # 259 features validation_1-auc:0.848588
# p = 69 # 238 features validation_1-auc:0.848547
# p = 67 # 247 features validation_1-auc:0.847925
# p = 65 # 240 features validation_1-auc:0.846769
# p = 60 # 222 features validation_1-auc:0.848581

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)

X_sel = X[features]

X_train, X_test, y_train, y_test = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.4)

# xgboost parameter tuning with p = 75
# recipe: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19083/best-practices-for-parameter-tuning-on-models/108783#post108783

ratio = float(np.sum(y == 1)) / np.sum(y==0)
# Initial parameters for the parameter exploration
# clf = xgb.XGBClassifier(missing=9999999999,
#                 max_depth = 10,
#                 n_estimators=1000,
#                 learning_rate=0.1, 
#                 nthread=4,
#                 subsample=1.0,
#                 colsample_bytree=0.5,
#                 min_child_weight = 5,
#                 scale_pos_weight = ratio,
#                 seed=4242)

# gives : validation_1-auc:0.845644
# max_depth=8 -> validation_1-auc:0.846341
# max_depth=6 -> validation_1-auc:0.845738
# max_depth=7 -> validation_1-auc:0.846504
# subsample=0.8 -> validation_1-auc:0.844440
# subsample=0.9 -> validation_1-auc:0.844746
# subsample=1.0,  min_child_weight=8 -> validation_1-auc:0.843393
# min_child_weight=3 -> validation_1-auc:0.848534
# min_child_weight=1 -> validation_1-auc:0.846311
# min_child_weight=4 -> validation_1-auc:0.847994
# min_child_weight=2 -> validation_1-auc:0.847934
# min_child_weight=3, colsample_bytree=0.3 -> validation_1-auc:0.847498
# colsample_bytree=0.7 -> validation_1-auc:0.846984
# colsample_bytree=0.6 -> validation_1-auc:0.847856
# colsample_bytree=0.5, learning_rate=0.05 -> validation_1-auc:0.847347
# max_depth=8 -> validation_1-auc:0.847352
# learning_rate = 0.07 -> validation_1-auc:0.847432
# learning_rate = 0.2 -> validation_1-auc:0.846444
# learning_rate = 0.15 -> validation_1-auc:0.846889
# learning_rate = 0.09 -> validation_1-auc:0.846680
# learning_rate = 0.1 -> validation_1-auc:0.847432
# max_depth=7 -> validation_1-auc:0.848534
# learning_rate = 0.05 -> validation_1-auc:0.847347
# 

clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 5,
                n_estimators=1000,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                reg_alpha=0.03,
                seed=1301)
                
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

test['n0'] = (test == 0).sum(axis=1)
# test['logvar38'] = test['var38'].map(np.log1p)
# # Encode var36 as category
# test['var36'] = test['var36'].astype('category')
# test = pd.get_dummies(test)
test_normalized = normalize(test, axis=0)
pca = PCA(n_components=2)
test_pca = pca.fit_transform(test_normalized)
test['PCA1'] = test_pca[:,0]
test['PCA2'] = test_pca[:,1]
sel_test = test[features]    
y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)