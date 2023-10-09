from __future__ import division
import numpy as np
import math as ma
import argparse
import time
import copy

def qt_reconstruct_b(tree,sq,f2,f3):
    f3_rec = np.copy(f2)
    Nl = len(sq)
    N_tau = np.size(tree,0)
    e_rec = np.zeros((N_tau,1))

    # calculate values of all possible leaves from acquired image
    a2 = -1*np.ones((N_tau,1))
    tree_rec = np.hstack([tree[0:N_tau,:], a2])
    #print('N_tau', N_tau)

    for k in range(1,N_tau+1):
        b = sq[Nl - int(tree[k-1,0])][:,int(tree[k-1,1])-1]
        f3x = f3[int(b[0]-1):int(b[2]),int(b[1]-1):int(b[3])]
        if tree[k-1,2] == 2: #acquire
            # print('tree acquire')
            m = np.mean(f3x)
            #print('f3x, m', np.shape(f3x), m)

            f3_rec[int(b[0])-1:int(b[2]),int(b[1])-1:int(b[3])] = m
            tree_rec[k-1,3] = m
            e_rec[k-1,0] = np.sum(np.abs(f3_rec[int(b[0]-1):int(b[2]),int(b[1]-1):int(b[3])] - m))
        else:
            e_rec[k-1,0] = np.sum(np.abs(f3_rec[int(b[0]-1):int(b[2]),int(b[1]-1):int(b[3])] - f3x))

    return tree_rec,f3_rec,e_rec
