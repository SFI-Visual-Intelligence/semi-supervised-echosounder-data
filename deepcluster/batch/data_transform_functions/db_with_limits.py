import numpy as np
from data.normalization import db


def db_with_limits_for_comparisonP2(data, labels):
    data = db(data)
    data[data>0] = 0
    data[data<-75] = -75
    return data, labels


def db_with_limits(data, labels, echogram, frequencies):
    data = db(data)
    data[data>0] = 0
    data[data<-75] = -75
    return data, labels, echogram, frequencies

def db_with_limits_img(data):
    data = db(data)
    data[data>0] = 0
    data[data<-75] = -75
    return data