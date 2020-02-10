
from data.normalization import db


def db_with_limits(data, labels, echogram, frequencies):
    data = db(data)
    data[data>0] = 0
    data[data<-75] = -75
    return data, labels, echogram, frequencies
