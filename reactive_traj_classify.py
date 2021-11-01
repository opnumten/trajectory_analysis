import numpy as np
import os
from os import listdir
import pandas as pd
from scipy.stats import kde
import seaborn as sns
import copy
from math import exp,log
import pickle
import scipy.ndimage as ndimage
import scipy.interpolate.fitpack as fitpack
from sklearn import manifold,decomposition,random_projection,cluster,metrics,preprocessing,mixture,model_selection
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
import scipy.io as sio
from statsmodels.tsa.ar_model import AR
from scipy import signal



def consecutive_arrs(data, stepsize=3):
    return np.split(data, np.where(np.diff(data)>stepsize)[0]+1)
 


def autocorr(x, range=144,step=1):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in np.arange(1, range,step)])

def find_reaction_start_end(traj_state,dwell_thres=6):
    reaction_start=traj_state.shape[0]
    reaction_end=0
    enter_M=0

    E_arrs=consecutive_arrs(np.where(traj_state==0)[0])
    M_arrs=consecutive_arrs(np.where(traj_state==2)[0])
    for m in range(len(E_arrs)):
        if E_arrs[m].shape[0]>dwell_thres:
            reaction_start=E_arrs[m][0]
            break
    n=len(M_arrs)
    while n>=1:
        if M_arrs[n-1].shape[0]>dwell_thres:
            reaction_end=M_arrs[n-1][-1]
            break
        n-=1
    for k in range(len(M_arrs)):
        if M_arrs[k].shape[0]>dwell_thres:
            enter_M=M_arrs[k][0]
            break
    return reaction_start,reaction_end,enter_M


def autocorr(x, lag_range,step=1):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in np.arange(1,lag_range,step)])

def sliding_window_ar1_coef(x,window_size,t_range,step=1):
    return np.array([(AR(x[t:t+window_size]).fit(1)).params[1] for t in np.arange(0,t_range,step)])


def ar1_tipping_time(reaction_traj,window_size_p=4,t_range_p=2):

    ar1_t=[]
    window_size=reaction_traj.shape[0]//window_size_p
    t_range=reaction_traj.shape[0]//t_range_p
    for j in range(reaction_traj.shape[1]):
        traj_ar1=sliding_window_ar1_coef(reaction_traj[:,j], window_size,t_range)

        peaks, _ = signal.find_peaks(traj_ar1)
        max_peak=peaks[np.argmax(traj_ar1[peaks])]
        ar1_t.append(max_peak+window_size)



        # ar1_t.append(np.argmax(traj_ar1)+window_size)
                
    return ar1_t

def cross_corr_delay(reaction_traj,d1=0,d2=1):
    n=reaction_traj.shape[0]
    y1=reaction_traj[:,d1]
    y2=reaction_traj[:,d2]
    corr0 = signal.correlate(y1, y2, mode='full') / np.sqrt(np.correlate(y1, y1)*np.correlate(y2, y2))

    corr_range=n//2
    corr=corr0[(n-corr_range)+1:corr_range+n]

#                 corr=abs(corr)
    if np.argmax(corr)>=corr_range:
        print('vim first',np.argmax(corr)-corr_range)

        cc_lag=np.argmax(corr)-corr_range
        max_corr=corr[np.argmax(corr)]
    else:
        print('morph first',-(corr_range-np.argmax(corr)))
        cc_lag=-(corr_range-np.argmax(corr))
        max_corr=corr[np.argmax(corr)]
    return cc_lag,max_corr

def find_intermediate_part(ls_score):
    trans_state=ls_score
    reactive_inds=[]

    P_arrs=consecutive_arrs(np.where(trans_state==1)[0])
    for m in range(len(P_arrs)):
        if P_arrs[m].shape[0]>0:
            if P_arrs[m][-1]<(ls_score.shape[0]-1) and P_arrs[m][0]>0:
                if trans_state[P_arrs[m][-1]+1]==2 and trans_state[P_arrs[m][0]-1]==0:
                    reactive_inds.append(P_arrs[m])


    return reactive_inds 

# def cluster_points(X, mu):
#     cluster_labels=np.zeros((X.shape[0]))
#     for x in X:
#         bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
#         cluster_labels[np.where(X==x)[0]]=bestmukey

#     return cluster_labels
# def sliding_window_ar1(x,window_size,step=1):
#     return np.array([np.corrcoef(x[t:t+window_size], x[t+1:t+1+window_size])[0,1] for t in np.arange(0,x.shape[0]-window_size-1,step)])
# def sliding_window_std(x,window_size,step=1):
#     return np.array([np.std(x[t:t+window_size]) for t in np.arange(0,x.shape[0]-window_size,step)]) 


