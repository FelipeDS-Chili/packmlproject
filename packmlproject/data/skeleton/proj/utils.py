import numpy as numpy
import time
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import os
import pandas as pd



def compute_precision_macro(y_pred, y_true):

    return precision_score(y_true, y_pred, average='macro')

def compute_precision(y_pred, y_true):


    return precision_score(y_true, y_pred)

def compute_f1_macro(y_pred, y_true):


    return precision_score(y_true, y_pred, average='macro')


def compute_recall(y_pred, y_true):

     return recall_score(y_true, y_pred)

def compute_f1(y_pred, y_true):

     return f1_score(y_true, y_pred)

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed


def get_data():

    pass




if __name__ == "__main__":
    companies = get_data()
    print(companies.head(10))
    print(companies.shape)
    print(companies.columns)
    print(companies.info())
    print(companies.state_code.value_counts())
    print(companies.country_code.value_counts())
    companies.to_csv("companies_test.csv")



