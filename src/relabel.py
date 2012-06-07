# Some code to be ported to cython ?

import numpy
def relabel(label, max_size= -1):
    """
    relabel limits the number of region in the label array. 
    They are ranked relatively to their size (number of pixels)
    
    @param label: a label array coming out of scipy.ndimage.measurement.label
    @param max_size: the max number of label wanted #Unused for now !
    @return array like label
    """
    max_label = label.max()
    count, pos = np.histogram(label, max_label)
    out = np.zeros_like(label)
    sortCount = count.argsort()

    f = lambda idx:sortCount[max_label - idx - 1]
#    for i in  range(max_size):
#        idx = sortCount[max_label - i - 1]
#        print i, max_label - i, sortCount[max_label - i - 1], idx, count[idx]
#        out[label == idx] = i

    return f(label)
