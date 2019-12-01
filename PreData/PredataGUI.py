import numpy as np
import random
from keras.utils import to_categorical
import sys
#from data_loader import Loader
sys.path.insert(1, '../Model')
from Model import deepsleepnet


def data_preparation(datafile=".npz"):
    with np.load(datafile) as npz:
        data_X = npz['x']
        data_y = npz['y']
    data_y = to_categorical(data_y)
    return data_X, data_y

def _model():
    pre_model = deepsleepnet.featurenet()
    dsn = deepsleepnet.deepsleepnet(pre_model)
    return pre_model, dsn
"""
def data_prepared_kflod(datafile = ".npz"):
    data = Loader()
    data.load_pretrain(fold = 0, path = 
    data_X = data.X_test
    data_X_seq = data.X_seq_test
    return data_X, data_X_seq
"""
