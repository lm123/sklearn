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
import time

def learn(data_home=None, remove_dup=False, reduce_dimension=False):
    """Apply BaggingClassifier algorithnm to fit and predict.

    Parameters
    ----------
    data_home : string, optional
        Specify the folder for the datasets.

    remove_dup : bool, default=False
        If True, remove duplicate lines with the same features.

    reduce_dimension : bool, default=False
        If True, reduce dimension.

    """
 
    if data_home == None:
        data_home = "."

    #fetch train datasets
    indoor_pos = fetch_indoor_pos(data_home=data_home, remove_dup=remove_dup)

    #fetch test datasets
    indoor_pos_test = fetch_indoor_pos(data_home=data_home, is_train=False, remove_dup=remove_dup)

    X = indoor_pos.data
    X_test = indoor_pos_test.data

    #dimension reduce processing
    if reduce_dimension == True:
        pca = PCA()
        pca.fit(indoor_pos.data)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.95) + 1
        print("reduced dimension:",d)
        pca = PCA(n_components = d)
        X = pca.fit_transform(X)
        X_test = pca.fit_transform(X_test)

    start = time.time()

    #predict x
    clf = BaggingClassifier(
              tree.DecisionTreeClassifier(), n_estimators=15,
              bootstrap=True, n_jobs=-1, oob_score=True)
    clf.fit(X, np.asarray(np.ravel(indoor_pos.target_x), dtype=np.int))
    pred_x = clf.predict(X_test)
    pred_x_model = clf.predict(indoor_pos.orig_data)

    #predict y
    clf = BaggingClassifier(
              tree.DecisionTreeClassifier(), n_estimators=15,
              bootstrap=True, n_jobs=-1, oob_score=True)
    clf.fit(X, np.asarray(np.ravel(indoor_pos.target_y), dtype=np.int))
    pred_y = clf.predict(X_test)
    pred_y_model = clf.predict(indoor_pos.orig_data)

    #write csv file
    with open('result_test.csv', mode='w') as w_file:
        csv_writer = csv.writer(w_file, delimiter=",")
        #write header
        csv_writer.writerow(['x', 'y', '2.1G(10)', '2.1G(11)', '2.1G(12)', '2.1G(4)', '2.1G(7)', '2.1G(8)', '3.5G(10)', '3.5G(11)', '3.5G(12)', '3.5G(4)', '3.5G(7)', '3.5G(8)'])

        for i in range(len(pred_x)):
          row = []
          row = np.append(row, pred_x[i])
          row = np.append(row, pred_y[i])
          row = np.append(row, indoor_pos_test.data[i])
          row[0] = int(row[0])
          row[1] = int(row[1])
          csv_writer.writerow(row)

    #write csv file
    with open('result_model.csv', mode='w') as w_file:
        csv_writer = csv.writer(w_file, delimiter=",")
        #write header
        csv_writer.writerow(['x', 'y', '2.1G(10)', '2.1G(11)', '2.1G(12)', '2.1G(4)', '2.1G(7)', '2.1G(8)', '3.5G(10)', '3.5G(11)', '3.5G(12)', '3.5G(4)', '3.5G(7)', '3.5G(8)'])

        for i in range(len(pred_x_model)):
          row = []
          row = np.append(row, pred_x_model[i])
          row = np.append(row, pred_y_model[i])
          row = np.append(row, indoor_pos.orig_data[i])
          row[0] = int(row[0])
          row[1] = int(row[1])
          csv_writer.writerow(row)

    done = time.time()

    print("time to predict:", done - start)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, np.asarray(np.ravel(indoor_pos.target_y), dtype=np.int), test_size=0.25, random_state=5)

    clf = BaggingClassifier(
              tree.DecisionTreeClassifier(), n_estimators=15,
              bootstrap=True, n_jobs=-1, oob_score=True)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    acc_bc = accuracy_score(y_test, y_test_pred)

    print("accuracy for predict:", acc_bc)

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
        print("Usage: indoorpospred.py datafile remove|not reduce|not")

