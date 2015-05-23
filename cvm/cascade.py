'''
Implementation of recursive cascading for SVM models

Based on:
Graf, Hans P., et al. "Parallel support vector machines: The cascade svm."
Advances in neural information processing systems. 2004.
'''

from __future__ import division

import numpy as np

def cascade(labeledPointRDD, reducer, nmax, iter_left=1, last_sv=-1):

    print "Entering Cascade: ", iter_left

    n = labeledPointRDD.count()
    numPartitions = int(2**(np.ceil(np.log(n / nmax)/np.log(2.0))))
    # numLevels = int(np.round(np.log(numPartitions) / np.log(2) + 1))
    leafsRDD = labeledPointRDD.repartition(numPartitions)

    # append last_sv to each partition (not the first time)
    append_sv = True if last_sv != -1 else False

    while numPartitions > 1:
        print 'Currently {} partitions left'.format(numPartitions)
        print 'Size of data: {}'.format(leafsRDD.count())

        numPartitions = int(numPartitions / 2)

        # need cache else lazy evaluation is killing
        if append_sv:
            reducer_sv = lambda data_iter: reducer(last_sv, data_iter)
            append_sv = False
        else:
            reducer_sv = lambda data_iter: reducer(-1, data_iter)

        leafsRDD = leafsRDD.mapPartitions(reducer_sv, True) \
                           .coalesce(numPartitions) \
                           .cache()

    new_sv = leafsRDD.collect()

    # compare last_sv to new_sv - if identical, we are done
    if iter_left == 0 or identical_SV(new_sv, last_sv):
        return new_sv

    return cascade(labeledPointRDD, reducer, nmax, iter_left-1, new_sv)

def identical_SV(sv1, sv2, epsilon=0.9):

    if sv2 == -1:
        return False

    sv1_f = set([tuple(elem.features) for elem in sv1])
    sv2_f = set([tuple(elem.features) for elem in sv2])

    return sv1_f == sv2_f






