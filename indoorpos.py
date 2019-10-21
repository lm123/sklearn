"""Code Rally 2019 dataset.

https://www.kaggle.com/hzhao011/code-rally-2019/

"""

import sys
import errno
import os
import numpy as np
from os.path import dirname, exists, join
from sklearn.datasets.base import get_data_home
from sklearn.utils import Bunch

def fetch_indoor_pos(data_home=None, is_train=True,
                     return_X_y=False, remove_dup=False):
    """Load the code rally 2019 dataset (classification).

    =================   ====================================
    Samples total       25811                             
    Dimensionality      12
    Features            continuous (float)
    =================   ====================================

    Parameters
    ----------
    data_home : string, optional
        Specify the folder for the datasets.

    is_train : bool, default=True
        Whether is train dataset.

    return_X_y : bool, default=False
        If True, returns ``(data, target_x, target_y, orig_data)`` instead of a Bunch object.
   
    remove_dup : bool, default=False
        If True, remove the duplicate lines with the same features, only keep one 
        against one series of features.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
         - 'data', the data to learn removing duplicate entries.
         - 'target_x', the regression target x for each sample.
         - 'target_y', the regression target y for each sample.
         - 'orig_data', the data to learn.

    (data, target_x, target_y) : tuple if ``return_X_y`` is True

    """
    data_home = get_data_home(data_home=data_home)
    indoorpos = _fetch_brute_indoor_pos(data_home=data_home, is_train = is_train, remove_dup=remove_dup)

    data = indoorpos.data
    target_x = indoorpos.target_x
    target_y = indoorpos.target_y
    orig_data = indoorpos.orig_data

    if return_X_y:
        return data, target_x, target_y, orig_data

    return Bunch(data=data, target_x=target_x, target_y=target_y, orig_data=orig_data)


def _fetch_brute_indoor_pos(data_home=None, is_train=True, remove_dup=False):

    """Load the indoor position dataset

    Parameters
    ----------
    data_home : string, optional
        Specify the folder for the datasets.

    Returns
    -------
    dataset : dict-like object with the following attributes:
        dataset.data : numpy array of shape (xxx, 12)
            Each row corresponds to the 12 features in the dataset.
        dataset.target : numpy array of shape (xxx,)
            Each value corresponds to x,y

    """

    data_home = get_data_home(data_home=data_home)

    if is_train:
       indoorpos_trainfile = join(data_home, "train.csv")
    else:
       indoorpos_trainfile = join(data_home, "test.csv")

    available = exists(indoorpos_trainfile)

    if available:
        dt = [('x', int),
              ('y', int),
              ('2.1G(10)', float),
              ('2.1G(11)', float),
              ('2.1G(12)', float),
              ('2.1G(4)', float),
              ('2.1G(7)', float),
              ('2.1G(8)', float),
              ('3.5G(10)', float),
              ('3.5G(11)', float),
              ('3.5G(12)', float),
              ('3.5G(4)', float),
              ('3.5G(7)', float),
              ('3.5G(8)', float)]

        DT = np.dtype(dt)

        file_ = open(indoorpos_trainfile, mode='r')
        Xy = []
        linenum = 0
        for line in file_.readlines():
            if linenum > 0:
                Xy.append(line.replace('\n', '').split(','))
            linenum = linenum + 1
        file_.close()

        Xy = np.asarray(Xy, dtype=object)

        if is_train:
          for j in range(len(dt)):
              Xy[:, j] = Xy[:, j].astype(DT[j])
        else:
          for j in range(2,len(dt)):
              Xy[:, j] = Xy[:, j].astype(DT[j])

        X = Xy[:,2:14]

        if not is_train:
            return Bunch(data=X, target_x=[], target_y=[], orig_data=X)

        dict = {}
        indexes = []
        X1 = [] 
        y1 = Xy[:, 0:1]
        y2 = Xy[:, 1:2]

        #duplicate entries processing
        for i in range(len(X)):
            if not remove_dup:
                indexes.append(i)
                X1.append(X[i])
                continue

            str1 = ""
            str1 = ''.join(str(e) for e in X[i])
            if str1 in dict:
                continue

            dict[str1] = i 
            indexes.append(i)
            X1.append(X[i])

        new_y1 = []
        new_y2 = []
        y11 = np.asarray(np.ravel(y1), dtype=np.int)
        y21 = np.asarray(np.ravel(y2), dtype=np.int)

        for i in indexes:
            new_y1 = np.append(new_y1, y11[i])
            new_y2 = np.append(new_y2, y21[i])

    elif not available:
        raise IOError("Data not found")

    return Bunch(data=X1, target_x=new_y1, target_y=new_y2, orig_data=X)
