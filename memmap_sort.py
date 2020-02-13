import numpy as np
list = np.loadtxt('memmaplist.txt', dtype=str)


def yearsort(list):
    y2014 = []
    y2015 = []
    y2016 = []
    y2017 = []
    for sample in list:
        if sample[:4] == '2014':
            y2014.append(sample)
        elif sample[:4] == '2015':
            y2015.append(sample)
        elif sample[:4] == '2016':
            y2016.append(sample)
        else:
           y2017.append(sample)
    y2014, y2015, y2016, y2017 = np.array(y2014), np.array(y2015), np.array(y2016), np.array(y2017)
    return y2014, y2015, y2016, y2017

def sanity_check(yearlist, year):
    for sample in yearlist:
        if sample[9:13] != year:
            print(sample[9:13])
            print(sample)

def sample_echograms(yearlist, size):
    idx = np.sort(np.random.choice(len(yearlist), size))
    print(idx)
    return yearlist[idx]

y2014, y2015, y2016, y2017 = yearsort(list)
sanity_check(y2014, '2014')
sanity_check(y2015, '2015')
sanity_check(y2016, '2016')
sanity_check(y2017, '2017')

size=10
y2014_sample = sample_echograms(y2014, size)
y2015_sample = sample_echograms(y2015, size)
y2016_sample = sample_echograms(y2016, size)
y2017_sample = sample_echograms(y2017, size)
