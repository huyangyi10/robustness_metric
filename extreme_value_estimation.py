#-*- coding = utf-8 -*-
#@Time : 2021-5-29 11:59
#@Author : CollionsHu
#@File : extreme_value_estimation.py
#@Software : PyCharm

import os
import sys
import glob
from functools import partial
from multiprocessing import Pool
import scipy
import scipy.io as sio
from scipy.stats import weibull_min
import scipy.optimize
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fmin_with_reg(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
                  full_output=0, disp=1, retall=0, callback=None, initial_simplex=None, shape_reg=0.01):
    # print('my optimier with shape regularizer = {}'.format(shape_reg))
    def func_with_reg(theta, x):
        shape = theta[2]
        log_likelyhood = func(theta, x)
        reg = shape_reg * shape * shape
        # penalize the shape parameter
        return log_likelyhood + reg

    return scipy.optimize.fmin(func_with_reg, x0, args, xtol, ftol, maxiter, maxfun,
                               full_output, disp, retall, callback, initial_simplex)


# fit using weibull_min.fit and run a K-S test
def fit_and_test(rescaled_sample, sample, loc_shift, shape_rescale, optimizer, c_i):
    [c, loc, scale] = weibull_min.fit(-rescaled_sample, c_i, optimizer=optimizer)
    loc = - loc_shift + loc * shape_rescale
    scale *= shape_rescale
    ks, pVal = scipy.stats.kstest(-sample, 'weibull_min', args=(c, loc, scale))
    return c, loc, scale, ks, pVal


def plot_weibull(sample, c, loc, scale, ks, pVal, p, q, figname="Lily_weibull_test.png"):
    # compare the sample histogram and fitting result
    fig, ax = plt.subplots(1, 1)

    x = np.linspace(-1.01 * max(sample), -0.99 * min(sample), 100);
    ax.plot(x, weibull_min.pdf(x, c, loc, scale), 'r-', label='fitted pdf ' + p + '-bnd')
    ax.hist(-sample, normed=True, bins=20, histtype='stepfilled')
    ax.legend(loc='best', frameon=False)
    plt.xlabel('-Lips_' + q)
    plt.ylabel('pdf')
    plt.title('c = {:.2f}, loc = {:.2f}, scale = {:.2f}, ks = {:.2f}, pVal = {:.2f}'.format(c, loc, scale, ks, pVal))
    plt.savefig(figname)
    plt.close()
    # model = figname.split("_")[1]
    # plt.savefig('./debug/'+model+'/'+figname)
    # plt.show() # can be used to pause the program


# We observe than the MLE estimator in scipy sometimes can converge to a bad
# value if the inital shape parameter c is too far from the true value. Thus we
# test a few different initializations and choose the one with best p-value all
# the initializations are tested in parallel; remove some of them to speedup
# computation.
# c_init = [0.01,0.1,0.5,1,5,10,20,50,70,100,200]
c_init = [0.1, 1, 5, 10, 20, 50, 100]


# Understood
def get_best_weibull_fit(sample, use_reg=False, shape_reg=0.01):
    # initialize dictionary to save the fitting results
    fitted_paras = {"c": [], "loc": [], "scale": [], "ks": [], "pVal": []}
    # reshape the data into a better range
    # this helps the MLE solver find the solution easier
    loc_shift = np.amax(sample)
    dist_range = np.amax(sample) - np.amin(sample)
    # if dist_range > 2.5:
    shape_rescale = dist_range
    # else:
    #     shape_rescale = 1.0
    print("shape rescale = {}".format(shape_rescale))
    rescaled_sample = np.copy(sample)
    rescaled_sample -= loc_shift
    rescaled_sample /= shape_rescale
    print("loc_shift = {}".format(loc_shift))
    ##print("rescaled_sample = {}".format(rescaled_sample))

    # fit weibull distn: sample follows reverse weibull dist, so -sample follows weibull distribution
    if use_reg:
        results = pool.map(partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale,
                                   partial(fmin_with_reg, shape_reg=shape_reg)), c_init)
    else:
        results = pool.map(
            partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale, scipy.optimize.fmin), c_init)

    for res, c_i in zip(results, c_init):
        c = res[0]
        loc = res[1]
        scale = res[2]
        ks = res[3]
        pVal = res[4]
        print(
            "[DEBUG][L2] c_init = {:5.5g}, fitted c = {:6.2f}, loc = {:7.2f}, scale = {:7.2f}, ks = {:4.2f}, pVal = {:4.2f}, max = {:7.2f}".format(
                c_i, c, loc, scale, ks, pVal, loc_shift))

        ## plot every fitted result
        # plot_weibull(sample,c,loc,scale,ks,pVal,p)

        fitted_paras['c'].append(c)
        fitted_paras['loc'].append(loc)
        fitted_paras['scale'].append(scale)
        fitted_paras['ks'].append(ks)
        fitted_paras['pVal'].append(pVal)

    # get the paras of best pVal among c_init
    max_pVal = np.nanmax(fitted_paras['pVal'])
    if np.isnan(max_pVal) or max_pVal < 0.001:
        print("ill-conditioned samples. Using maximum sample value.")
        # handle the ill conditioned case
        return -1, -1, -max(sample), -1, -1, -1

    max_pVal_idx = fitted_paras['pVal'].index(max_pVal)

    c_init_best = c_init[max_pVal_idx]
    c_best = fitted_paras['c'][max_pVal_idx]
    loc_best = fitted_paras['loc'][max_pVal_idx]
    scale_best = fitted_paras['scale'][max_pVal_idx]
    ks_best = fitted_paras['ks'][max_pVal_idx]
    pVal_best = fitted_paras['pVal'][max_pVal_idx]

    return c_init_best, c_best, loc_best, scale_best, ks_best, pVal_best


def get_extreme_value_estimate(G_max, norm="L2", figname="", use_reg=False, shape_reg=0.01):
    global plot_res
    c_init, c, loc, scale, ks, pVal = get_best_weibull_fit(G_max, use_reg, shape_reg)

    if norm == "L1":
        p = "i";
        q = "1"
    elif norm == "L2":
        p = "2";
        q = "2"
    elif norm == "Li":
        p = "1";
        q = "i"
    else:
        print("Lipschitz norm is not in 1, 2, i!")

    figname = figname + '_' + "L" + p + ".png"

    if plot_res is not None:
        plot_res.get()

    # plot_weibull(G_max,c,loc,scale,ks,pVal,p,q,figname)
    if figname:
        plot_res = pool.apply_async(plot_weibull, (G_max, c, loc, scale, ks, pVal, p, q, figname))

    return {'Lips_est': -loc, 'shape': c, 'loc': loc, 'scale': scale, 'ks': ks, 'pVal': pVal}

pool = Pool(processes = args['nthreads'])
