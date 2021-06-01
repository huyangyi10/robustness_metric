#-*- coding = utf-8 -*-
#@Time : 2021-5-29 11:38
#@Author : CollionsHu
#@File : uniform_sample_over_sphere.py
#@Software : PyCharm


import numpy as np
import math
import time
# from shmemarray import ShmemRawArray, NpShmemArray
from multiprocessing import Pool, current_process, cpu_count
from scipy.special import gammainc
from functools import partial

# generate random signs
def randsign(N):
    n_bytes = (N + 7) // 8
    rbytes = np.random.randint(0, 255, dtype=np.uint8, size=n_bytes)
    return (np.unpackbits(rbytes)[:N] - 0.5) * 2

def l2_samples(m, n, r, center):
    X = np.random.randn(m, n)
    s2 = np.sum(X * X, axis = 1)
    wantedArray = X * (np.tile(1.0*np.power(gammainc(n/2,s2/2), 1/n) / np.sqrt(s2), (n,1))).T
    sum = 0
    sum_min = 1
    for i in range(0, m):
        for j in range(0, n):
            sum = sum + wantedArray[i, j]*wantedArray[i, j]  #l2 norm
            wantedArray[i, j] = wantedArray[i, j] * r + center[j]
        # print(sum)
        if sum < sum_min:
            sum_min = sum
        sum = 0
    # print(sum_min)
    # print(m)
    return wantedArray

def linf_samples(m, n, r, center):
    wantedArray = np.random.uniform(-1.0, 1.0, (m,n))
    for i in range(0, m):
        for j in range(0, n):
            # sum = sum + newArray[i, j]*newArray[i, j]  #l2 norm
            wantedArray[i, j] = wantedArray[i, j] * r + center[j]
    return wantedArray

def l1_samples(m, n, r, center):
    # U is uniform random between 0, 1
    U = np.random.uniform(0, 1.0, (m,n-1))
    V = np.empty(shape=(m,n+1))
    # V is sorted U, with 0 and 1 added to the begin and the end
    V[:,0] = 0.0
    V[:,-1] = 1.0
    V[:,1:-1] = np.sort(U)
    # X is the interval between each V_i
    X = V[:,1:] - V[:,:-1]
    # randomly flip the sign of each X
    s = randsign(m * n).reshape(m,n)
    # generate random samples over the surface of unit sphere
    wantedArray = X * s
    # sum = 0
    for i in range(0, m):
        ran = np.random.rand()
        for j in range(0, n):
            # sum = sum + newArray[i, j]*newArray[i, j]  #l2 norm
            wantedArray[i, j] = wantedArray[i, j] * ran
            # sum = sum + abs(wantedArray[i, j])

        # print(sum)
        # print(np.append(newArray[i], math.sqrt(sum)))
        # newArray[i] = np.concatenate((newArray[i], [sum]))
        # newArray[i] = np.append(newArray[i], sum)
        # sum = 0

    for i in range(0, m):
        for j in range(0, n):
            # sum = sum + newArray[i, j]*newArray[i, j]  #l2 norm
            wantedArray[i, j] = wantedArray[i, j] * r + center[j]

        # print(sum)
        # print(np.append(newArray[i], math.sqrt(sum)))
        # newArray[i] = np.concatenate((newArray[i], [sum]))
        # newArray[i] = np.append(newArray[i], sum)

    return wantedArray

def uniformsampleoverspheretogether(m, n, norm, r, center):
     if norm == "l2":
         samples = l2_samples(m, n, r, center)
         return samples
     elif norm == "l1":
         samples = l1_samples(m, n, r, center)
         return samples
     elif norm == "li":
         samples = linf_samples(m, n, r, center)
         return samples

# m stands for the number of batch, n for the sample number in each batch, r is the radius
def uniformsampleoversphere(m, n, norm, r, center):
    # paralell running
    partialuniformsample = partial(uniformsampleoverspheretogether, norm = norm, n = n, r = r, center = center)
    pool = Pool(10)
    result = pool.map(partialuniformsample, [int(m/10) for i in range(10)])
    return result

if __name__ == "__main__":
    result = uniformsampleoversphere(100, 3, "l2", 1, np.array([1.0, 2.0, 3.0]))
    print(result)

    # pool.map_async(partialuniformsample, [100 for i in range(10)])
    # samples = uniformsampleoversphere("l2", 100000, 15000, 1)
    # print(r.get() for r in res)
