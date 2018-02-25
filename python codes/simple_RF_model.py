print("Importing necessary libraries...")
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.ensemble import RandomForestClassifier

print("Importing training and test data...")
df_train = pd.read_csv("../input/train.csv")
df_test  = pd.read_csv("../input/test.csv")   

print("Defining training and test sets...")
id_test = df_test['ID']
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values
X_test = df_test.drop(['ID'], axis=1).values

print("Applying the method...")
clf = RandomForestClassifier(n_estimators=500, max_depth=5)

print("Fitting the model and making predictions...")
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

print("Cross validating and checking the score...")
scores = cv.cross_val_score(clf, X_train, y_train, cv=10) 
print(scores.mean())

print("Making submission...")
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)