from indoorpos import fetch_indoor_pos
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv

def learn(data_home=None, remove_dup=True, reduce_dimension=True):

    if data_home == None:
        data_home = "."

    indoor_pos = fetch_indoor_pos(data_home=data_home, remove_dup=remove_dup)
    indoor_pos_test = fetch_indoor_pos(data_home=data_home, is_train=False, remove_dup=remove_dup)

    X = indoor_pos.data
    X_test = indoor_pos_test.data
    if reduce_dimension == True:
        pca = PCA()
        pca.fit(indoor_pos.data)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.95) + 1
        print(d)
        pca = PCA(n_components = d)
        X = pca.fit_transform(X)
        X_test = pca.fit_transform(X_test)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, np.asarray(np.ravel(indoor_pos.target_x), dtype=np.int))
    pred_x = clf.predict(X_test)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, np.asarray(np.ravel(indoor_pos.target_y), dtype=np.int))
    pred_y = clf.predict(X_test)

    with open('result_dt.csv', mode='w') as w_file:
        csv_writer = csv.writer(w_file, delimiter=",")
        for i in range(len(pred_x)):
          row = []
          row = np.append(row, pred_x[i])
          row = np.append(row, pred_y[i])
          row = np.append(row, indoor_pos_test.data[i])
          row[0] = int(row[0])
          row[1] = int(row[1])
          csv_writer.writerow(row)

    clf = BaggingClassifier(
              tree.DecisionTreeClassifier(), n_estimators=15,
              bootstrap=True, n_jobs=-1, oob_score=True)
    clf.fit(X, np.asarray(np.ravel(indoor_pos.target_x), dtype=np.int))
    pred_x = clf.predict(X_test)

    clf = BaggingClassifier(
              tree.DecisionTreeClassifier(), n_estimators=15,
              bootstrap=True, n_jobs=-1, oob_score=True)
    clf.fit(X, np.asarray(np.ravel(indoor_pos.target_y), dtype=np.int))
    pred_y = clf.predict(X_test)

    with open('result_bc.csv', mode='w') as w_file:
        csv_writer = csv.writer(w_file, delimiter=",")
        for i in range(len(pred_x)):
          row = []
          row = np.append(row, pred_x[i])
          row = np.append(row, pred_y[i])
          row = np.append(row, indoor_pos_test.data[i])
          row[0] = int(row[0])
          row[1] = int(row[1])
          csv_writer.writerow(row)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, np.asarray(np.ravel(indoor_pos.target_y), dtype=np.int), test_size=0.25, random_state=5)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    acc_dt = accuracy_score(y_test, y_test_pred)
    
    clf = BaggingClassifier(
              tree.DecisionTreeClassifier(), n_estimators=15,
              bootstrap=True, n_jobs=-1, oob_score=True)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    acc_bc = accuracy_score(y_test, y_test_pred)
    
    print(acc_dt)
    print(acc_bc)

if __name__ == "__main__":
    from sys import argv
    
    if len(argv) == 4:
        remove_dup = False
        if argv[2] == "remove":
            remove_dup = True
     
        reduce_dimension = False
        if argv[3] == "reduce":
            reduce_dimension = True

        learn(argv[1], remove_dup=remove_dup, reduce_dimension=reduce_dimension)
    else:
        learn()

