 import pandas as pd
 import numpy as np
 from sklearn.cross_validation import train_test_split
 from sklearn.metrics import roc_auc_score
 from sklearn.metrics import accuracy_score 
 from sklearn.ensemble import ExtraTreesClassifier
 from sklearn.feature_selection import SelectFromModel
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import classification_report
 import matplotlib.pyplot as plt
 train = pd.read_csv("train.csv")
 test = pd.read_csv("test.csv")
 train = pd.read_csv("train.csv")
 test = pd.read_csv("test.csv")
 # clean and split data^M
 # remove constant columns (std = 0)^M
 remove = []
 for col in train.columns:
     if train[col].std() == 0:
         remove.append(col)
 train.drop(remove, axis=1, inplace=True)
 test.drop(remove, axis=1, inplace=True)
 # remove duplicated columns^M
 remove = []
 cols = train.columns
 for i in range(len(cols)-1):
     v = train[cols[i]].values
     for j in range(i+1,len(cols)):
         if np.array_equal(v,train[cols[j]].values):
             remove.append(cols[j])
 train.drop(remove, axis=1, inplace=True)
 test.drop(remove, axis=1, inplace=True)
 # split data into train and test^M
 test_id = test.ID
 test = test.drop(["ID"],axis=1)
 X = train.drop(["TARGET","ID"],axis=1)
 y = train.TARGET.values
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)
 print(X_train.shape, X_test.shape, test.shape)
 ## # Feature selection
 clf = ExtraTreesClassifier(random_state=1729)
 selector = clf.fit(X_train, y_train)
 feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)
 feat_imp[:40].plot(kind='bar', title='Feature Importances according to ExtraTreesClassifier', figsize=(12, 8))
 plt.ylabel('Feature Importance Score')
 plt.subplots_adjust(bottom=0.3)
 plt.savefig('1.png')
 plt.show()
 # clf.feature_importances_ 
 fs = SelectFromModel(selector, prefit=True)
 X_train = fs.transform(X_train)
 X_test = fs.transform(X_test)
 test = fs.transform(test)
 pivot = int(0.8 * len(X_train))
 X_train = np.array(X_train)
y = np.array(y)
validation_features = X_train[pivot:]
y_train = np.array(y_train)
y_test = np.array(y_test)
X_test = np.array(X_test)
print("Len validation:", len(X_test),"Len train:", len(X_train))
model = LogisticRegression()
model.fit(X_train, y_train)
test_m_a = model.score(X_test, y_test)
print(test_m_a)
train_m_a = model.score(X_train, y_train)
print(train_m_a)
 score=accuracy_score(y_test, model.predict(X_test))
 roc=roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
 cr=classification_report(y_test, model.predict(X_test))
 print(score)
 print(roc)
 print(cr)
